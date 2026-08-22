import json
import queue
import threading

import sounddevice as sd
from vosk import Model, KaldiRecognizer


SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 4000

SMALL_MODEL_PATH = "vosk_models/vosk-model-small-en-us-0.15"
LARGE_MODEL_PATH = "vosk_models/vosk-model-en-us-0.22"


class SpeechInputEngine:

    def __init__(self, small_keywords):

        # ==================================================
        # Result queues
        # ==================================================

        self.small_final_queue = queue.Queue()
        self.small_partial_queue = queue.Queue()

        self.large_final_queue = queue.Queue()
        self.large_partial_queue = queue.Queue()

        # ==================================================
        # Audio queues
        # ==================================================

        # The microphone callback puts audio here.
        self.audio_queue = queue.Queue(maxsize=100)

        # Each recognizer gets its own queue.
        self.small_audio_queue = queue.Queue(maxsize=100)
        self.large_audio_queue = queue.Queue(maxsize=100)

        # ==================================================
        # Model states
        # ==================================================

        self.small_enabled = False
        self.large_enabled = False

        self.running = False

        self.lock = threading.Lock()

        # ==================================================
        # Partial result tracking
        # ==================================================

        self.last_small_partial = ""
        self.last_large_partial = ""

        # ==================================================
        # Load models
        # ==================================================

        print("Loading small Vosk model...")

        self.small_model = Model(
            SMALL_MODEL_PATH
        )

        print("Loading large Vosk model...")

        self.large_model = Model(
            LARGE_MODEL_PATH
        )

        # ==================================================
        # Recognizers
        # ==================================================

        self.small_recognizer = KaldiRecognizer(
            self.small_model,
            SAMPLE_RATE,
            json.dumps(small_keywords)
        )

        self.large_recognizer = KaldiRecognizer(
            self.large_model,
            SAMPLE_RATE
        )

        # ==================================================
        # Audio stream
        # ==================================================

        self.stream = None

        # ==================================================
        # Worker threads
        # ==================================================

        self.small_thread = None
        self.large_thread = None

    # ======================================================
    # MICROPHONE CALLBACK
    # ======================================================

    def _audio_callback(
        self,
        indata,
        frames,
        time,
        status
    ):
        """
        This callback MUST remain extremely fast.

        No Vosk processing happens here.
        """

        if status:
            print("Audio status:", status)

        if not self.running:
            return

        data = bytes(indata)

        # Put the same audio into both recognizer queues.

        with self.lock:
            small_enabled = self.small_enabled
            large_enabled = self.large_enabled

        if small_enabled:
            try:
                self.small_audio_queue.put_nowait(data)
            except queue.Full:
                pass

        if large_enabled:
            try:
                self.large_audio_queue.put_nowait(data)
            except queue.Full:
                pass

    # ======================================================
    # SMALL MODEL WORKER
    # ======================================================

    def _small_worker(self):

        while self.running:

            try:
                data = self.small_audio_queue.get(
                    timeout=0.1
                )

            except queue.Empty:
                continue

            try:

                with self.lock:
                    enabled = self.small_enabled
                    recognizer = self.small_recognizer

                if not enabled:
                    continue

                if recognizer.AcceptWaveform(data):

                    result = json.loads(
                        recognizer.Result()
                    )

                    text = result.get(
                        "text",
                        ""
                    ).strip()

                    if text:
                        self.small_final_queue.put(
                            text
                        )

                    self.last_small_partial = ""

                else:

                    result = json.loads(
                        recognizer.PartialResult()
                    )

                    text = result.get(
                        "partial",
                        ""
                    ).strip()

                    if (
                        text
                        and text != self.last_small_partial
                    ):
                        self.small_partial_queue.put(
                            text
                        )

                        self.last_small_partial = text

            finally:
                self.small_audio_queue.task_done()

    # ======================================================
    # LARGE MODEL WORKER
    # ======================================================

    def _large_worker(self):

        while self.running:

            try:
                data = self.large_audio_queue.get(
                    timeout=0.1
                )

            except queue.Empty:
                continue

            try:

                with self.lock:
                    enabled = self.large_enabled
                    recognizer = self.large_recognizer

                if not enabled:
                    continue

                if recognizer.AcceptWaveform(data):

                    result = json.loads(
                        recognizer.Result()
                    )

                    text = result.get(
                        "text",
                        ""
                    ).strip()

                    if text:
                        self.large_final_queue.put(
                            text
                        )

                    self.last_large_partial = ""

                else:

                    result = json.loads(
                        recognizer.PartialResult()
                    )

                    text = result.get(
                        "partial",
                        ""
                    ).strip()

                    if (
                        text
                        and text != self.last_large_partial
                    ):
                        self.large_partial_queue.put(
                            text
                        )

                        self.last_large_partial = text

            finally:
                self.large_audio_queue.task_done()

    # ======================================================
    # START
    # ======================================================

    def start(self):

        if self.running:
            return

        self.running = True

        # ----------------------------------------------
        # Start small worker
        # ----------------------------------------------

        self.small_thread = threading.Thread(
            target=self._small_worker,
            daemon=True
        )

        self.small_thread.start()

        # ----------------------------------------------
        # Start large worker
        # ----------------------------------------------

        self.large_thread = threading.Thread(
            target=self._large_worker,
            daemon=True
        )

        self.large_thread.start()

        # ----------------------------------------------
        # Start microphone
        # ----------------------------------------------

        self.stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype="int16",
            channels=CHANNELS,
            callback=self._audio_callback,
        )

        self.stream.start()

        print("Microphone started.")

    # ======================================================
    # STOP
    # ======================================================

    def stop(self):

        if not self.running:
            return

        self.running = False

        with self.lock:
            self.small_enabled = False
            self.large_enabled = False

        # ----------------------------------------------
        # Stop microphone
        # ----------------------------------------------

        if self.stream is not None:

            self.stream.stop()
            self.stream.close()

            self.stream = None

        # ----------------------------------------------
        # Stop workers
        # ----------------------------------------------

        if self.small_thread is not None:

            self.small_thread.join(
                timeout=2
            )

            self.small_thread = None

        if self.large_thread is not None:

            self.large_thread.join(
                timeout=2
            )

            self.large_thread = None

        # ----------------------------------------------
        # Clear queues
        # ----------------------------------------------

        self._clear_audio_queues()

        print("Microphone stopped.")

    # ======================================================
    # CLEAR AUDIO QUEUES
    # ======================================================

    def _clear_audio_queues(self):

        self._clear_queue(
            self.small_audio_queue
        )

        self._clear_queue(
            self.large_audio_queue
        )

    @staticmethod
    def _clear_queue(q):

        while True:

            try:
                q.get_nowait()
                q.task_done()

            except queue.Empty:
                break

    # ======================================================
    # SMALL MODEL
    # ======================================================

    def small_on(self):

        with self.lock:
            self.small_enabled = True

        print("Small model ON.")

    def small_off(self):

        with self.lock:
            self.small_enabled = False

        self.last_small_partial = ""

        self._clear_queue(
            self.small_audio_queue
        )

        print("Small model OFF.")

    # ======================================================
    # CHANGE SMALL KEYWORDS
    # ======================================================

    def set_small_keywords(self, keywords):
        """
        Replace the small model grammar.

        The Vosk Model stays loaded.
        Only the KaldiRecognizer is recreated.
        """

        new_recognizer = KaldiRecognizer(
            self.small_model,
            SAMPLE_RATE,
            json.dumps(keywords)
        )

        with self.lock:

            self.small_recognizer = new_recognizer

            self.last_small_partial = ""

        # Remove audio and results from the old grammar.
        self._clear_queue(
            self.small_audio_queue
        )

        self._clear_small_results()

    def _clear_small_results(self):

        self._clear_queue(
            self.small_final_queue
        )

        self._clear_queue(
            self.small_partial_queue
        )

    # ======================================================
    # LARGE MODEL
    # ======================================================

    def large_on(self):

        # Create a fresh recognizer every time the large
        # model starts listening.

        new_recognizer = KaldiRecognizer(
            self.large_model,
            SAMPLE_RATE
        )

        with self.lock:

            self.large_recognizer = new_recognizer

            self.large_enabled = True

            self.last_large_partial = ""

        # Don't process audio captured before we started.
        self._clear_queue(
            self.large_audio_queue
        )

        self._clear_large_results()

        print("Large model ON.")

    def large_off(self):

        with self.lock:
            self.large_enabled = False

            self.last_large_partial = ""

        self._clear_queue(
            self.large_audio_queue
        )

        print("Large model OFF.")

    def _clear_large_results(self):

        self._clear_queue(
            self.large_final_queue
        )

        self._clear_queue(
            self.large_partial_queue
        )

    # ======================================================
    # SMALL RESULTS
    # ======================================================

    def get_small(self):

        try:
            return self.small_final_queue.get_nowait()

        except queue.Empty:
            return None

    def get_small_partial(self):

        try:
            return self.small_partial_queue.get_nowait()

        except queue.Empty:
            return None

    # ======================================================
    # LARGE RESULTS
    # ======================================================

    def get_large(self):

        try:
            return self.large_final_queue.get_nowait()

        except queue.Empty:
            return None

    def get_large_partial(self):

        try:
            return self.large_partial_queue.get_nowait()

        except queue.Empty:
            return None

    # ======================================================
    # PAUSE
    # ======================================================

    def pause(self):

        with self.lock:

            previous_state = {
                "small": self.small_enabled,
                "large": self.large_enabled,
            }

            self.small_enabled = False
            self.large_enabled = False

        self._clear_audio_queues()

        self.last_small_partial = ""
        self.last_large_partial = ""

        return previous_state

    # ======================================================
    # RESUME
    # ======================================================

    def resume(self, previous_state):

        self._clear_audio_queues()

        with self.lock:

            self.small_enabled = previous_state[
                "small"
            ]

            self.large_enabled = previous_state[
                "large"
            ]

        self.last_small_partial = ""
        self.last_large_partial = ""
