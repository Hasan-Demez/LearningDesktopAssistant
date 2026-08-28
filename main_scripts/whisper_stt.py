import queue
import threading
import time
from collections import deque

import numpy as np

from ten_vad import TenVad
from faster_whisper import WhisperModel


class WhisperInputEngine:
    """Self-contained Faster-Whisper + TEN VAD speech-to-text engine.

    Signals (main.py reads / controls these):
        on_final     — queue.Queue  — final transcribed text
        pause_event  — threading.Event — cleared to pause, set to resume
        stop_event   — threading.Event — set to shut down worker
    """

    SAMPLE_RATE = 16000

    def __init__(
        self,
        audio,
        model_path,
        device="cpu",
        compute_type="int8",
        language="en",
        silence_duration=0.8,
        end_grace_duration=0.8,
        min_speech_duration=0.1,
        max_utterance_duration=15.0,
        vad_hop_size=256,
        vad_threshold=0.5,
        preroll_duration=0.4,
        initial_prompt=None,
    ):

        self.audio = audio

        self.audio_queue = (
            self.audio.register("whisperx")
        )

        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self.language = language

        self.silence_duration = silence_duration

        # Short window after the VAD flag drops before
        # we declare the utterance finished. Absorbs
        # single-frame VAD flickers mid-sentence.
        self.end_grace_duration = end_grace_duration
        self.min_speech_duration = min_speech_duration
        self.max_utterance_duration = (
            max_utterance_duration
        )

        self.vad_hop_size = vad_hop_size
        self.vad_threshold = vad_threshold

        # ==================================================
        # PRE-ROLL BUFFER
        #
        # Rolling window of recent audio kept even while
        # no speech is flagged. When speech starts, this
        # is prepended so word onsets that TEN VAD
        # detected late are not chopped off.
        # ==================================================

        self.preroll_duration = preroll_duration

        preroll_samples = int(
            self.SAMPLE_RATE * preroll_duration
        )

        self.preroll_buffer = deque(
            maxlen=max(
                1,
                preroll_samples // self.vad_hop_size,
            )
        )

        # ==================================================
        # INITIAL PROMPT
        #
        # Primes Whisper's vocabulary toward the phrases
        # and jargon used most often. Reduces
        # hallucination on short/quiet utterances.
        # ==================================================

        if initial_prompt:

            self.initial_prompt = str(
                initial_prompt
            ).strip()

        else:

            self.initial_prompt = None

        # ==================================================
        # RESULTS
        # ==================================================

        self.final_queue = queue.Queue()

        # ==================================================
        # STATE
        # ==================================================

        self.running = False
        self.enabled = False

        self.lock = threading.Lock()

        # ==================================================
        # SIGNALS (controlled by main.py)
        # ==================================================

        self.pause_event = threading.Event()
        self.pause_event.set()

        self.stop_event = threading.Event()

        # ==================================================
        # CALLBACKS (set by main.py)
        #
        # on_speech_start() — fired the instant TEN VAD
        #     flags the beginning of speech. Used to cut
        #     off TTS playback immediately.
        #
        # on_speech_end() — fired when TEN VAD drops the
        #     flag (end of utterance boundary).
        # ==================================================

        self.on_speech_start = None
        self.on_speech_end = None

        # ==================================================
        # WORKER
        # ==================================================

        self.worker_thread = None

        # ==================================================
        # SPEECH STATE
        # ==================================================

        self.in_speech = False

        self.speech_buffer = []

        self.speech_start_time = None
        self.last_voice_time = None

        # ==================================================
        # TEN VAD
        # ==================================================

        self.vad = TenVad(
            hop_size=vad_hop_size,
            threshold=vad_threshold,
        )

        # ==================================================
        # FASTER WHISPER
        # ==================================================

        import os

        if not os.path.isdir(model_path):

            raise FileNotFoundError(
                f"Faster-Whisper model directory does not "
                f"exist:\n{model_path}"
            )

        print(
            f"Loading local Faster-Whisper model:\n"
            f"  {model_path}"
        )

        self.model = WhisperModel(
            model_path,
            device=device,
            compute_type=compute_type,
            local_files_only=True,
        )

        print(
            "Faster-Whisper loaded."
        )

    # ======================================================
    # OUTPUT PROPERTY
    # ======================================================

    @property
    def on_final(self):
        """Final transcribed text queue."""
        return self.final_queue

    # ======================================================
    # WORKER
    # ======================================================

    def _worker(self):

        while not self.stop_event.is_set():

            try:

                data = self.audio_queue.get(
                    timeout=0.1
                )

            except queue.Empty:

                continue

            try:

                # ------------------------------------------
                # Check pause signal
                # ------------------------------------------

                if not self.pause_event.is_set():

                    continue

                with self.lock:

                    enabled = self.enabled

                if not enabled:
                    continue

                self._process_audio(
                    data
                )

            except Exception as e:

                print(
                    f"Whisper input error: {e}"
                )

            finally:

                self.audio_queue.task_done()

    # ======================================================
    # AUDIO PROCESSING
    # ======================================================

    def _process_audio(self, audio):

        samples = np.frombuffer(
            audio,
            dtype=np.int16
        ).astype(
            np.float32
        ) / 32768.0

        if samples.ndim > 1:

            samples = samples.mean(
                axis=1
            )

        # Convert to int16 for TEN VAD
        audio_int16 = (
            samples * 32768
        ).astype(np.int16)

        # Process with TEN VAD
        # TEN VAD expects hop_size samples per call
        # We need to process in chunks of hop_size
        hop_size = self.vad_hop_size

        for i in range(
            0,
            len(audio_int16),
            hop_size,
        ):

            chunk = audio_int16[
                i : i + hop_size
            ]

            if len(chunk) < hop_size:
                break

            probability, flag = self.vad.process(
                chunk
            )

            now = time.monotonic()

            # ------------------------------------------
            # Always keep recent audio in the pre-roll
            # ring buffer so speech onsets detected
            # late by the VAD can be recovered.
            # ------------------------------------------

            self.preroll_buffer.append(
                samples[i : i + hop_size]
            )

            # ------------------------------------------
            # Speech detected (flag == 1)
            # ------------------------------------------

            if flag == 1:

                if not self.in_speech:

                    self.in_speech = True

                    self.speech_start_time = now
                    self.last_voice_time = now

                    print(
                        f"[VAD] SPEECH START "
                        f"(prob={probability:.2f})"
                    )

                    # ----------------------------------
                    # Fire callback immediately so TTS
                    # can be cut off mid-utterance.
                    # ----------------------------------

                    if self.on_speech_start:

                        try:

                            self.on_speech_start()

                        except Exception as e:

                            print(
                                f"[VAD] speech_start "
                                f"callback error: {e}"
                            )

                    # ----------------------------------
                    # Start the utterance with the
                    # pre-roll buffer so the beginning
                    # of the first word is included.
                    # ----------------------------------

                    self.speech_buffer = list(
                        self.preroll_buffer
                    )

                else:

                    self.speech_buffer.append(
                        samples[
                            i : i + hop_size
                        ]
                    )

                    self.last_voice_time = now

                # ------------------------------------------
                # Maximum utterance
                # ------------------------------------------

                if (
                    self.speech_start_time is not None
                    and
                    now - self.speech_start_time
                    >= self.max_utterance_duration
                ):

                    print("[VAD] MAX DURATION reached.")

                    self._finish_utterance()

                    return

            # ------------------------------------------
            # No speech (flag == 0)
            #
            # The utterance ends when TEN VAD itself
            # drops the flag — we only keep a short
            # grace window to absorb tiny VAD flickers,
            # not a long silence timer.
            # ------------------------------------------

            else:

                if self.in_speech:

                    self.speech_buffer.append(
                        samples[
                            i : i + hop_size
                        ]
                    )

                    if (
                        self.last_voice_time is not None
                        and
                        now - self.last_voice_time
                        >= self.end_grace_duration
                    ):

                        print(
                            f"[VAD] SPEECH END "
                            f"(duration="
                            f"{now - self.speech_start_time:.2f}s)"
                        )

                        if self.on_speech_end:

                            try:

                                self.on_speech_end()

                            except Exception as e:

                                print(
                                    f"[VAD] speech_end "
                                    f"callback error: {e}"
                                )

                        self._finish_utterance()

    # ======================================================
    # FINISH UTTERANCE
    # ======================================================

    def _finish_utterance(self):

        if not self.speech_buffer:

            self._reset_speech_state()

            return

        audio = np.concatenate(
            self.speech_buffer
        )

        duration = (
            len(audio)
            / self.SAMPLE_RATE
        )

        self._reset_speech_state()

        if duration < self.min_speech_duration:

            return

        self._transcribe(
            audio
        )

    # ======================================================
    # RESET
    # ======================================================

    def _reset_speech_state(self):

        self.in_speech = False

        self.speech_buffer = []

        self.preroll_buffer.clear()

        self.speech_start_time = None
        self.last_voice_time = None

    # ======================================================
    # TRANSCRIBE WITH FASTER WHISPER
    # ======================================================

    def _transcribe(self, audio):

        try:

            # Faster-whisper's internal VAD (Silero ONNX)
            # requires float32 input, so pass the raw
            # float samples directly.
            audio_float32 = (
                audio.astype(np.float32)
            )

            segments, info = self.model.transcribe(
                audio_float32,
                language=self.language,
                beam_size=5,
                vad_filter=True,
                vad_parameters={
                    "threshold": 0.5,
                    "min_speech_duration_ms": 100,
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 200,
                },
                condition_on_previous_text=False,
                initial_prompt=self.initial_prompt,
            )

            texts = []

            for segment in segments:

                text = segment.text.strip()

                if text:

                    texts.append(
                        text
                    )

            text = " ".join(
                texts
            ).strip()

            if not text:
                return

            print(
                "WHISPER:",
                text
            )

            self.final_queue.put(
                text
            )

        except Exception as e:

            print(
                f"Faster-Whisper transcription error: "
                f"{e}"
            )

    # ======================================================
    # START
    # ======================================================

    def start(self):

        if self.running:
            return

        self.running = True

        self.worker_thread = (
            threading.Thread(
                target=self._worker,
                daemon=True,
            )
        )

        self.worker_thread.start()

        print(
            "Whisper started."
        )

    # ======================================================
    # STOP
    # ======================================================

    def stop(self):

        if not self.running:
            return

        self.running = False
        self.enabled = False
        self.stop_event.set()

        if self.worker_thread is not None:

            self.worker_thread.join(
                timeout=2
            )

            self.worker_thread = None

        self._reset_speech_state()

        self._clear_audio_queue()

        print(
            "Whisper stopped."
        )

    # ======================================================
    # ON
    # ======================================================

    def on(self):

        with self.lock:

            self.enabled = True

        self._clear_audio_queue()

        self._clear_results()

        self._reset_speech_state()

        print(
            "Whisper ON."
        )

    def large_on(self):

        self.on()

    # ======================================================
    # OFF
    # ======================================================

    def off(self):

        with self.lock:

            self.enabled = False

        self._reset_speech_state()

        self._clear_audio_queue()

        self._clear_results()

        print(
            "Whisper OFF."
        )

    def large_off(self):

        self.off()

    # ======================================================
    # STATUS
    # ======================================================

    def is_on(self):

        with self.lock:

            return self.enabled

    # ======================================================
    # GET FINAL TEXT
    # ======================================================

    def get(self):

        try:

            return self.final_queue.get_nowait()

        except queue.Empty:

            return None

    # ======================================================
    # CLEAR AUDIO
    # ======================================================

    def _clear_audio_queue(self):

        while True:

            try:

                self.audio_queue.get_nowait()

                self.audio_queue.task_done()

            except queue.Empty:

                break

    # ======================================================
    # CLEAR RESULTS
    # ======================================================

    def _clear_results(self):

        while True:

            try:

                self.final_queue.get_nowait()

                self.final_queue.task_done()

            except queue.Empty:

                break