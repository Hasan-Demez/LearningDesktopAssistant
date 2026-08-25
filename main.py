import os
import time

# ===========================================================
# THREAD OPTIMIZATION (must be before any heavy imports)
#
# Set OpenMP/MKL thread counts to match physical cores.
# This prevents PyTorch from using too few or too many threads.
# ===========================================================

# _NUM_THREADS = os.cpu_count() or 4
_NUM_THREADS = 4
os.environ.setdefault("OMP_NUM_THREADS", str(_NUM_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_NUM_THREADS))

import torch
torch.set_num_threads(_NUM_THREADS)
torch.set_num_interop_threads(1)

from main_scripts.audio_input import SharedAudioInput
from main_scripts.audio_output import play_beep
from main_scripts.stt import SpeechInputEngine
from main_scripts.whisper_stt import WhisperInputEngine

from main_scripts.tts import SpeechOutputEngine
from main_scripts.kokoro_tts import KokoroSpeechOutputEngine

from main_scripts.llm import DeskLLM
from main_scripts.commands import AssistantController
from main_scripts.setup_models import get_model_paths

_START_TIME = time.monotonic()


def _ts():
    """Seconds since program start."""
    return time.monotonic() - _START_TIME


def _load(name, fn):
    """Run fn(), print timing, return result."""
    t0 = _ts()
    result = fn()
    print(
        f"[{_ts():.3f}s] {name} loaded "
        f"({_ts() - t0:.1f}s)"
    )
    return result


# ==========================================================
# MODEL PATHS (single source of truth)
# ==========================================================

paths = get_model_paths()


# ==========================================================
# AUDIO INPUT
# ==========================================================

audio = SharedAudioInput(
    sample_rate=16000,
    channels=1,
    block_size=1600,
)


# ==========================================================
# SMALL VOSK
# ==========================================================

speech_input = SpeechInputEngine(
    model_path=paths["vosk"],
    small_keywords=[
        "wake up",
        "buddy",
        "[unk]",
    ],
    audio=audio,
)


# ==========================================================
# WHISPERX
# ==========================================================

whisper_input = WhisperInputEngine(
    audio=audio,
    model_path=paths["whisper"],
    device="cpu",
    compute_type="int8",
    language="en",
    silence_duration=1.5,
    min_speech_duration=0.3,
    energy_threshold=0.01,
    max_utterance_duration=15.0,
)


# ==========================================================
# SYSTEM TTS (pyttsx3 — commands, status, wake/sleep)
# ==========================================================

speech_output = SpeechOutputEngine(
    voice_index=0,
    rate=200,
    volume=1.0,
)


# ==========================================================
# LLM TTS (Kokoro ONNX — LLM responses)
# ==========================================================

llm_tts = KokoroSpeechOutputEngine(
    model_path=paths["kokoro_onnx_model"],
    voices_path=paths["kokoro_onnx_voices"],
    voice="af_heart",
    lang_code="en-us",
    speed=1.0,
    volume=1.0,
)


# ==========================================================
# LLM
# ==========================================================

llm = DeskLLM(
    model_path=paths["llm"],
    n_ctx=1024,
    n_threads=3,
)


# ==========================================================
# CONTROLLER
# ==========================================================

controller = AssistantController(
    speech_input=speech_input,
    speech_output=speech_output,
    llm=llm,
    assistant_name="buddy",
)


# ==========================================================
# SIGNAL WIRING
#
# main.py is the only place that knows about all
# components.  STT engines are paused whenever TTS
# is speaking OR speech is queued OR the LLM is
# still generating.
# ==========================================================

def _update_stt_pause():
    """Pause STT when any audio output is active."""
    should_pause = (
        llm.is_generating.is_set()
        or llm_tts.speech_queued.is_set()
        or speech_output.speaking.is_set()
    )

    if should_pause:
        speech_input.pause_event.clear()
        whisper_input.pause_event.clear()
    else:
        speech_input.pause_event.set()
        whisper_input.pause_event.set()

    return should_pause


# ==========================================================
# CONTROLLER SIGNAL HANDLERS
#
# Whisper should only be on when in chatting/listening/talking
# modes, not when just awake (idle).
# ==========================================================

def _on_mode_change(mode, awake):
    """Handle mode changes from controller.

    Whisper is enabled only for chatting/listening/talking modes.
    """
    if mode in ("chatting", "listening", "talking"):
        whisper_input.large_on()
    else:
        whisper_input.large_off()


def _on_state_change(awake):
    """Handle state changes from controller."""
    if not awake:
        whisper_input.large_off()


# Register signal handlers
controller.on_mode_change(_on_mode_change)
controller.on_state_change(_on_state_change)


# ==========================================================
# STARTUP
# ==========================================================

print(f"[{_ts():.3f}s] Starting...")

t_total = _ts()

audio.start()

speech_input.start()
whisper_input.start()

speech_input.set_small_keywords(
    controller.sleep_keywords()
)

speech_input.small_on()
whisper_input.large_off()

print(
    f"[{_ts():.3f}s] All systems ready "
    f"({_ts() - t_total:.1f}s)"
)

speech_output.talk(
    "Turned on."
)


# ==========================================================
# MAIN LOOP
# ==========================================================

_ready_beep_played = False

try:

    while True:

        # ==================================================
        # UPDATE STT PAUSE STATE + BEEP ON READY
        # ==================================================

        paused = _update_stt_pause()

        if not paused and not _ready_beep_played:
            play_beep()
            _ready_beep_played = True
        elif paused:
            _ready_beep_played = False

        # ==================================================
        # SMALL VOSK — commands
        # ==================================================

        small_text = speech_input.get_small()

        if small_text:

            controller.handle(small_text)

        # ==================================================
        # WHISPERX — speech for LLM
        # ==================================================

        llm_text = whisper_input.get()

        if llm_text:

            print(
                f"[{_ts():.3f}s] "
                f"WHISPERX: {llm_text}"
            )

            llm.submit(llm_text)

        # ==================================================
        # LLM SENTENCES → KOKORO
        # ==================================================

        sentence = llm.get_sentence()

        if sentence:

            print(
                f"[{_ts():.3f}s] "
                f"BUDDY: {sentence}"
            )

            llm_tts.talk(sentence)

except KeyboardInterrupt:

    pass


finally:

    try:

        speech_output.talk(
            "Turned off."
        )

        speech_output.wait_until_done()

    except Exception:
        pass

    try:

        llm_tts.wait_until_done()

    except Exception:
        pass

    speech_input.stop()
    whisper_input.stop()

    audio.stop()

    speech_output.stop()
    llm_tts.stop()

    llm.stop()