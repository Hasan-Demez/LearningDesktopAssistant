import queue
import threading

import sounddevice as sd


class SharedAudioInput:
    """
    One physical microphone stream shared by multiple consumers.

    Example:

        microphone
            |
            +--> small_vosk
            |
            +--> whisper

    Each consumer gets its own queue, so a slow consumer
    does not block the microphone callback.
    """

    SAMPLE_RATE = 16000
    CHANNELS = 1
    BLOCK_SIZE = 1600       # 100 ms

    def __init__(
        self,
        sample_rate=SAMPLE_RATE,
        channels=CHANNELS,
        block_size=BLOCK_SIZE,
        dtype="int16",
        queue_size=100,
    ):

        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size
        self.dtype = dtype
        self.queue_size = queue_size

        # ==================================================
        # STATE
        # ==================================================

        self.running = False

        self.lock = threading.Lock()

        # ==================================================
        # CONSUMERS
        # ==================================================

        self.consumers = {}

        # ==================================================
        # STREAM
        # ==================================================

        self.stream = None

    # ======================================================
    # REGISTER CONSUMER
    # ======================================================

    def register(self, name):

        with self.lock:

            if name in self.consumers:
                raise ValueError(
                    f"Audio consumer already exists: "
                    f"{name}"
                )

            self.consumers[name] = queue.Queue(
                maxsize=self.queue_size
            )

        print(
            f"Audio consumer registered: {name}"
        )

        return self.consumers[name]

    # ======================================================
    # REMOVE CONSUMER
    # ======================================================

    def unregister(self, name):

        with self.lock:

            q = self.consumers.pop(
                name,
                None
            )

        if q is not None:

            self._clear_queue(q)

            print(
                f"Audio consumer removed: {name}"
            )

    # ======================================================
    # GET QUEUE
    # ======================================================

    def get_queue(self, name):

        with self.lock:

            if name not in self.consumers:

                raise KeyError(
                    f"Unknown audio consumer: "
                    f"{name}"
                )

            return self.consumers[name]

    # ======================================================
    # MICROPHONE CALLBACK
    # ======================================================

    def _audio_callback(
        self,
        indata,
        frames,
        time_info,
        status,
    ):

        if status:

            print(
                "Audio status:",
                status
            )

        if not self.running:
            return

        # Copy immediately because PortAudio reuses
        # the callback buffer.
        data = bytes(indata)

        # --------------------------------------------------
        # Snapshot consumers under lock.
        # --------------------------------------------------

        with self.lock:

            queues = list(
                self.consumers.values()
            )

        # --------------------------------------------------
        # Distribute without ever blocking callback.
        # --------------------------------------------------

        for q in queues:

            try:

                q.put_nowait(data)

            except queue.Full:

                # Drop audio for this consumer rather
                # than blocking the microphone callback.
                pass

    # ======================================================
    # START
    # ======================================================

    def start(self):

        if self.running:
            return

        self.running = True

        self.stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            dtype=self.dtype,
            channels=self.channels,
            callback=self._audio_callback,
        )

        self.stream.start()

        print(
            "Shared microphone started."
        )

    # ======================================================
    # STOP
    # ======================================================

    def stop(self):

        if not self.running:
            return

        self.running = False

        # --------------------------------------------------
        # Stop stream
        # --------------------------------------------------

        if self.stream is not None:

            try:

                self.stream.stop()

            except Exception:
                pass

            try:

                self.stream.close()

            except Exception:
                pass

            self.stream = None

        # --------------------------------------------------
        # Clear all queues
        # --------------------------------------------------

        with self.lock:

            queues = list(
                self.consumers.values()
            )

        for q in queues:

            self._clear_queue(q)

        print(
            "Shared microphone stopped."
        )

    # ======================================================
    # CLEAR CONSUMER
    # ======================================================

    def clear(self, name):

        q = self.get_queue(name)

        self._clear_queue(q)

    # ======================================================
    # CLEAR ALL
    # ======================================================

    def clear_all(self):

        with self.lock:

            queues = list(
                self.consumers.values()
            )

        for q in queues:

            self._clear_queue(q)

    # ======================================================
    # QUEUE HELPER
    # ======================================================

    @staticmethod
    def _clear_queue(q):

        while True:

            try:

                q.get_nowait()

                try:
                    q.task_done()
                except ValueError:
                    pass

            except queue.Empty:

                break

    # ======================================================
    # STATUS
    # ======================================================

    def is_running(self):

        return self.running