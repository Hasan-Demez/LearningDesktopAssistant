import queue
import threading
import time
import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import whisperx


class WhisperInputEngine:
    """Self-contained WhisperX speech-to-text engine.

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
        min_speech_duration=0.3,
        energy_threshold=0.015,
        max_utterance_duration=15.0,
        batch_size=1,
        vad_onset=0.5,
        vad_offset=0.363,
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
        self.min_speech_duration = min_speech_duration
        self.energy_threshold = energy_threshold
        self.max_utterance_duration = (
            max_utterance_duration
        )

        self.batch_size = batch_size
        self.vad_onset = vad_onset
        self.vad_offset = vad_offset

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
        # WHISPERX
        # ==================================================

        self.model = None

        import os

        if not os.path.isdir(model_path):

            raise FileNotFoundError(
                f"WhisperX model directory does not "
                f"exist:\n{model_path}"
            )

        print(
            f"Loading local WhisperX model:\n"
            f"  {model_path}"
        )

        self.model = whisperx.load_model(
            model_path,
            device=device,
            device_index=0,
            compute_type=compute_type,
            language=language,
            local_files_only=True,
        )

        print(
            "WhisperX loaded."
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

        rms = np.sqrt(
            np.mean(
                samples * samples
            )
        )

        now = time.monotonic()

        # ==================================================
        # ALREADY IN SPEECH
        # ==================================================

        if self.in_speech:

            self.speech_buffer.append(
                samples
            )

            if rms >= self.energy_threshold:

                self.last_voice_time = now

            # ----------------------------------------------
            # Maximum utterance
            # ----------------------------------------------

            if (
                self.speech_start_time is not None
                and
                now - self.speech_start_time
                >= self.max_utterance_duration
            ):

                self._finish_utterance()

                return

            # ----------------------------------------------
            # End of speech
            # ----------------------------------------------

            if (
                self.last_voice_time is not None
                and
                now - self.last_voice_time
                >= self.silence_duration
            ):

                self._finish_utterance()

            return

        # ==================================================
        # CURRENTLY SILENT
        # ==================================================

        if rms >= self.energy_threshold:

            self.in_speech = True

            self.speech_start_time = now
            self.last_voice_time = now

            self.speech_buffer = [
                samples
            ]

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

        self.speech_start_time = None
        self.last_voice_time = None

    # ======================================================
    # TRANSCRIBE WITH WHISPERX
    # ======================================================

    def _transcribe(self, audio):

        try:

            result = self.model.transcribe(
                audio,
                batch_size=self.batch_size,
                language=self.language,
            )

            segments = result.get(
                "segments",
                []
            )

            texts = []

            for segment in segments:

                text = segment.get(
                    "text",
                    ""
                ).strip()

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
                "WHISPERX:",
                text
            )

            self.final_queue.put(
                text
            )

        except Exception as e:

            print(
                f"WhisperX transcription error: "
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
            "WhisperX started."
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
            "WhisperX stopped."
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
            "WhisperX ON."
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
            "WhisperX OFF."
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