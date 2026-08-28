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
from main_scripts.commands import VoskController, WhisperController
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
# WHISPER (Faster-Whisper + TEN VAD)
# ==========================================================

whisper_input = WhisperInputEngine(
    audio=audio,
    model_path=paths["whisper"],
    device="cpu",
    compute_type="int8",
    language="en",
    silence_duration=0.5,
    end_grace_duration=0.8,
    min_speech_duration=0.1,
    max_utterance_duration=15.0,
    vad_hop_size=256,
    vad_threshold=0.5,
    # initial_prompt=(
    #     "The user is talking to a desktop voice assistant "
    #     "named Buddy. Common spoken commands are: "
    #     "\"pause buddy\", \"continue buddy\", \"stop buddy\". "
    #     "Conversations are short everyday sentences."
    # ),
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
# VOSK CONTROLLER (wake/sleep/chat mode transitions)
# ==========================================================

vosk_controller = VoskController(
    speech_input=speech_input,
    speech_output=speech_output,
    llm=llm,
    assistant_name="buddy",
)


# ==========================================================
# WHISPER CONTROLLER (pause/continue/stop during chatting)
# ==========================================================

whisper_controller = WhisperController(
    llm=llm,
    llm_tts=llm_tts,
    speech_output=speech_output,
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
    """Pause STT when any audio output is active.

    Vosk is paused when LLM is generating or Kokoro is speaking.
    Faster Whisper is paused when LLM is generating (but TEN VAD is always active).
    TEN VAD is never paused during chatting mode.
    """
    should_pause_vosk = (
        llm.is_generating.is_set()
        or llm_tts.speech_queued.is_set()
        or speech_output.speaking.is_set()
    )

    should_pause_whisper = (
        llm.is_generating.is_set()
    )

    if should_pause_vosk:
        speech_input.pause_event.clear()
    else:
        speech_input.pause_event.set()

    if should_pause_whisper:
        whisper_input.pause_event.clear()
    else:
        whisper_input.pause_event.set()

    return should_pause_vosk or should_pause_whisper


# ==========================================================
# VOSK CONTROLLER SIGNAL HANDLERS
#
# Whisper should only be on when in chatting mode.
# ==========================================================

def _on_vosk_mode_change(mode, awake):
    """Handle mode changes from Vosk controller.

    Whisper is enabled only for chatting mode.
    """
    if mode == "chatting":
        whisper_input.large_on()
    else:
        whisper_input.large_off()


def _on_vosk_state_change(awake):
    """Handle state changes from Vosk controller."""
    if not awake:
        whisper_input.large_off()


# Register Vosk controller signal handlers
vosk_controller.on_mode_change(_on_vosk_mode_change)
vosk_controller.on_state_change(_on_vosk_state_change)


# ==========================================================
# WHISPER CONTROLLER SIGNAL HANDLERS
#
# Handle pause/continue/stop commands during chatting.
# ==========================================================

def _on_whisper_mode_change(mode, awake):
    """Handle mode changes from Whisper controller."""
    if mode == "idle":
        whisper_input.large_off()


def _on_whisper_pause_change(paused):
    """Handle pause state changes from Whisper controller."""
    if paused:
        llm_tts.pause()
    else:
        llm_tts.resume()


# ==========================================================
# TEN VAD BARGE-IN
#
# Fired the instant TEN VAD detects the START of speech
# (no transcription delay).  If Kokoro is talking, cut it
# off mid-utterance so the user is never talked over.
# ==========================================================

def _on_vad_speech_start():
    """Cut off Kokoro the moment speech is detected.

    Barge-in is a hard cut: the current utterance and
    everything still queued is dropped — nothing is
    stashed or replayed afterwards.
    """

    if (
        llm_tts.speaking.is_set()
        or llm_tts.speech_queued.is_set()
    ):

        print(
            f"[{_ts():.3f}s] BARGE-IN: speech detected "
            f"→ cutting off Kokoro."
        )

        llm_tts.flush()


whisper_input.on_speech_start = _on_vad_speech_start


# Register Whisper controller signal handlers
whisper_controller.on_mode_change(_on_whisper_mode_change)
whisper_controller.on_pause_change(_on_whisper_pause_change)


# ==========================================================
# STARTUP
# ==========================================================

print(f"[{_ts():.3f}s] Starting...")

t_total = _ts()

audio.start()

speech_input.start()
whisper_input.start()

speech_input.set_small_keywords(
    vosk_controller.sleep_keywords()
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
        # SMALL VOSK — commands (only when not chatting)
        # ==================================================

        small_text = speech_input.get_small()

        if small_text:

            vosk_controller.handle(small_text)

        # ==================================================
        # WHISPER — speech for LLM (only when chatting)
        # ==================================================

        llm_text = whisper_input.get()

        if llm_text:

            print(
                f"[{_ts():.3f}s] "
                f"WHISPER: {llm_text}"
            )

            # Check if it's a Whisper command
            if whisper_controller.handle(llm_text):
                # Command was handled (pause/continue/stop)
                pass
            else:
                # Not a command, submit to LLM
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