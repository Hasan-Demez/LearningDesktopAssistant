import json
import queue
import threading

from vosk import Model, KaldiRecognizer


SAMPLE_RATE = 16000


class SpeechInputEngine:
    """Self-contained Vosk keyword detector.

    Signals (main.py reads / controls these):
        on_final   — queue.Queue  — final recognized text
        on_partial — queue.Queue  — partial recognized text
        pause_event — threading.Event — cleared to pause, set to resume
        stop_event  — threading.Event — set to shut down worker
    """

    def __init__(
        self,
        model_path,
        small_keywords,
        audio,
    ):

        self.audio = audio

        self.small_audio_queue = (
            self.audio.register("small_vosk")
        )

        # ==================================================
        # STATE
        # ==================================================

        self.small_enabled = False

        self.running = False

        self.lock = threading.Lock()

        # ==================================================
        # PARTIAL RESULT
        # ==================================================

        self.last_small_partial = ""

        # ==================================================
        # LOAD MODEL
        # ==================================================

        print(
            "Loading small Vosk model..."
        )

        self.small_model = Model(
            model_path
        )

        # ==================================================
        # RECOGNIZER
        # ==================================================

        self.small_recognizer = (
            KaldiRecognizer(
                self.small_model,
                SAMPLE_RATE,
                json.dumps(small_keywords),
            )
        )

        # ==================================================
        # OUTPUT QUEUES (read by main.py)
        # ==================================================

        self.small_final_queue = queue.Queue()
        self.small_partial_queue = queue.Queue()

        # ==================================================
        # SIGNALS (controlled by main.py)
        # ==================================================

        self.pause_event = threading.Event()
        self.pause_event.set()

        self.stop_event = threading.Event()

        # ==================================================
        # WORKER
        # ==================================================

        self.small_thread = None

    # ======================================================
    # OUTPUT PROPERTIES
    # ======================================================

    @property
    def on_final(self):
        """Final recognized text queue."""
        return self.small_final_queue

    @property
    def on_partial(self):
        """Partial recognized text queue."""
        return self.small_partial_queue

    # ======================================================
    # SMALL VOSK WORKER
    # ======================================================

    def _small_worker(self):

        while not self.stop_event.is_set():

            try:

                data = self.small_audio_queue.get(
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
                        and
                        text
                        != self.last_small_partial
                    ):

                        self.small_partial_queue.put(
                            text
                        )

                        self.last_small_partial = text

            finally:

                self.small_audio_queue.task_done()

    # ======================================================
    # START
    # ======================================================

    def start(self):

        if self.running:
            return

        self.running = True

        self.small_thread = threading.Thread(
            target=self._small_worker,
            daemon=True,
        )

        self.small_thread.start()

        print(
            "Small Vosk microphone started."
        )

    # ======================================================
    # STOP
    # ======================================================

    def stop(self):

        if not self.running:
            return

        self.running = False
        self.stop_event.set()

        with self.lock:
            self.small_enabled = False

        if self.small_thread is not None:

            self.small_thread.join(
                timeout=2
            )

            self.small_thread = None

        self._clear_audio_queue()

        print(
            "Small Vosk stopped."
        )

    # ======================================================
    # CLEAR QUEUE
    # ======================================================

    def _clear_audio_queue(self):

        while True:

            try:

                self.small_audio_queue.get_nowait()

                self.small_audio_queue.task_done()

            except queue.Empty:

                break

    # ======================================================
    # SMALL MODEL
    # ======================================================

    def small_on(self):

        with self.lock:

            self.small_enabled = True

        print(
            "Small model ON."
        )

    def small_off(self):

        with self.lock:

            self.small_enabled = False

            self.last_small_partial = ""

        self._clear_audio_queue()

        self._clear_small_results()

        print(
            "Small model OFF."
        )

    # ======================================================
    # CHANGE KEYWORDS
    # ======================================================

    def set_small_keywords(self, keywords):

        new_recognizer = (
            KaldiRecognizer(
                self.small_model,
                SAMPLE_RATE,
                json.dumps(keywords),
            )
        )

        with self.lock:

            self.small_recognizer = (
                new_recognizer
            )

            self.last_small_partial = ""

        self._clear_audio_queue()

        self._clear_small_results()

    # ======================================================
    # CLEAR RESULTS
    # ======================================================

    def _clear_small_results(self):

        self._clear_queue(
            self.small_final_queue
        )

        self._clear_queue(
            self.small_partial_queue
        )

    # ======================================================
    # GET RESULTS (called by main.py)
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

    @staticmethod
    def _clear_queue(q):

        while True:

            try:

                q.get_nowait()
                q.task_done()

            except queue.Empty:

                break