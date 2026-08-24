import queue
import threading
import time

import numpy as np
import sounddevice as sd

from kokoro_onnx import Kokoro


class KokoroSpeechOutputEngine:
    """ONNX-optimized Kokoro TTS with parallel pipeline.

    Architecture:
        text_queue -> generate_worker -> audio_queue -> player_worker -> speaker

    While the player plays sentence N, the generator pre-renders
    sentence N+1 so there is minimal gap between utterances.

    Signals (main.py reads / controls these):
        speaking     — threading.Event — set while audio is playing
        done         — threading.Event — set when fully drained
        speech_queued — threading.Event — set while anything in pipeline
    """

    SAMPLE_RATE = 24000

    def __init__(
        self,
        model_path: str,
        voices_path: str,
        voice: str = "af_heart",
        lang_code: str = "en-us",
        speed: float = 1.0,
        volume: float = 1.0,
    ):

        self.model_path = model_path
        self.voices_path = voices_path
        self.voice = voice
        self.lang_code = lang_code

        self.speed = speed
        self.volume = volume

        # ==================================================
        # QUEUES (two-stage pipeline)
        # ==================================================

        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()

        self.running = True

        # ==================================================
        # SIGNALS (read by main.py)
        # ==================================================

        self.speaking = threading.Event()
        self.done = threading.Event()
        self.speech_queued = threading.Event()

        # ==================================================
        # STATE
        # ==================================================

        self._pending_count = 0
        self._count_lock = threading.Lock()

        self.kokoro = None

        # ==================================================
        # LOAD MODEL
        # ==================================================

        self._load_kokoro()

        # ==================================================
        # WORKERS
        # ==================================================

        self._gen_thread = threading.Thread(
            target=self._generate_worker,
            daemon=True,
        )

        self._player_thread = threading.Thread(
            target=self._player_worker,
            daemon=True,
        )

        self._gen_thread.start()
        self._player_thread.start()

    # ======================================================
    # LOAD KOKORO
    # ======================================================

    def _load_kokoro(self):

        try:

            self.kokoro = Kokoro(
                self.model_path,
                self.voices_path,
            )

        except Exception as e:

            raise RuntimeError(
                "Failed to load Kokoro ONNX:\n"
                f"{e}"
            ) from e

    # ======================================================
    # PUBLIC API
    # ======================================================

    def talk(self, text: str):

        if not self.running:
            return

        if not text:
            return

        text = str(text).strip()

        if not text:
            return

        with self._count_lock:

            self._pending_count += 1

            self.done.clear()

            self.speech_queued.set()

        self.text_queue.put(text)

    # ======================================================
    # GENERATE WORKER
    #
    # Pulls text from text_queue, runs ONNX inference,
    # puts pre-rendered audio on audio_queue.
    # ======================================================

    def _generate_worker(self):

        while self.running:

            text = self.text_queue.get()

            if text is None:

                # Poison pill — pass it to player too
                self.audio_queue.put(None)

                self.text_queue.task_done()

                break

            try:

                samples, sample_rate = (
                    self.kokoro.create(
                        text,
                        voice=self.voice,
                        speed=self.speed,
                        lang=self.lang_code,
                    )
                )

                if self.volume != 1.0:

                    samples = samples * self.volume

                samples = np.asarray(
                    samples, dtype=np.float32
                )

                self.audio_queue.put(
                    (samples, sample_rate)
                )

            except Exception as e:

                print(
                    f"[KOKORO] Generate error: {e}"
                )

                # Put silence so player doesn't hang
                self.audio_queue.put(
                    (np.zeros(1, dtype=np.float32),
                     self.SAMPLE_RATE)
                )

            finally:

                self.text_queue.task_done()

    # ======================================================
    # PLAYER WORKER
    #
    # Pulls pre-rendered audio from audio_queue,
    # writes to speakers.
    # ======================================================

    def _player_worker(self):

        while self.running:

            item = self.audio_queue.get()

            if item is None:

                # Nothing left to play
                with self._count_lock:

                    self._pending_count = 0

                    self.speech_queued.clear()

                    self.done.set()

                self.audio_queue.task_done()

                break

            samples, sample_rate = item

            try:

                self.speaking.set()

                stream = sd.OutputStream(
                    samplerate=sample_rate,
                    channels=1,
                    dtype="float32",
                )

                stream.start()
                stream.write(samples)
                stream.stop()
                stream.close()

            except Exception as e:

                print(
                    f"[KOKORO] Play error: {e}"
                )

            finally:

                self.speaking.clear()

                self.audio_queue.task_done()

                with self._count_lock:

                    self._pending_count -= 1

                    if self._pending_count <= 0:

                        self._pending_count = 0

                        self.speech_queued.clear()

                        self.done.set()

    # ======================================================
    # STATUS
    # ======================================================

    def is_speaking(self):

        return self.speaking.is_set()

    @property
    def is_playing(self):

        return self.speaking.is_set()

    # ======================================================
    # WAIT
    # ======================================================

    def wait_until_done(self):

        self.done.wait()

    # ======================================================
    # STOP
    # ======================================================

    def stop(self):

        if not self.running:
            return

        self.running = False

        # Send poison pills
        self.text_queue.put(None)

        self._gen_thread.join(timeout=2)
        self._player_thread.join(timeout=2)

        self.kokoro = None