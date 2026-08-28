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
        # PAUSE (barge-in support)
        #
        # pause_event — cleared to pause playback
        #     mid-utterance, set to allow playback.
        #
        # Pausing is a hard cut: whatever remains of the
        # current sentence is dropped, not stashed.
        # resume() simply unpauses so future sentences
        # can play again.
        # ==================================================

        self.pause_event = threading.Event()
        self.pause_event.set()   # not paused by default

        # Incremented by flush() so an in-flight player
        # knows its item was cancelled and must not
        # stash a resume buffer afterwards.
        self._epoch = 0

        # ==================================================
        # PLAYBACK CHUNKING
        #
        # Audio is written to the speaker in small chunks
        # (25 ms) so a pause can take effect almost
        # instantly instead of after a whole sentence.
        # ==================================================

        self._chunk_samples = int(
            self.SAMPLE_RATE * 0.025
        )

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
    # writes to speakers in small chunks so playback
    # can be cut off mid-utterance (barge-in).
    #
    # When paused, the remaining samples are dropped.
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

            # Snapshot the epoch so flush() during
            # playback can invalidate this item.
            epoch = self._epoch

            try:

                self.speaking.set()

                stream = sd.OutputStream(
                    samplerate=sample_rate,
                    channels=1,
                    dtype="float32",
                )

                stream.start()

                # --------------------------------------
                # Chunked playback loop.
                #
                # Between chunks we check pause_event:
                #   - cleared → stop writing immediately
                #     and drop the remainder (hard cut).
                #   - set → keep going.
                # --------------------------------------

                offset = 0

                total = len(samples)

                while offset < total:

                    if (
                        not self.pause_event.is_set()
                        or epoch != self._epoch
                    ):

                        print(
                            f"[KOKORO] Cut off at "
                            f"{offset}/{total} "
                            f"samples "
                            f"({offset / sample_rate:.2f}s "
                            f"of "
                            f"{total / sample_rate:.2f}s) — "
                            "remainder dropped."
                        )

                        break

                    end = min(
                        offset + self._chunk_samples,
                        total,
                    )

                    chunk = samples[offset:end]

                    stream.write(chunk)

                    offset = end

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
    # PAUSE / RESUME (barge-in)
    # ======================================================

    def pause(self):
        """Pause playback immediately.

        The remainder of the current utterance is
        dropped — nothing is stashed for replay.
        """
        self.pause_event.clear()

        print("[KOKORO] pause() requested.")

    def resume(self):
        """Unpause so future sentences can play.

        Does NOT replay anything that was cut off
        while paused — that audio is gone.
        """
        self.pause_event.set()

        print("[KOKORO] resume() requested.")

    def flush(self):
        """Drop everything: queued text, rendered
        audio, and any in-flight partial utterance.
        """

        # Invalidate any in-flight playback so the
        # player stops immediately.
        self._epoch += 1

        # Make sure playback is unpaused so future
        # sentences play normally.
        self.pause_event.set()

        while True:

            try:

                self.text_queue.get_nowait()
                self.text_queue.task_done()

            except queue.Empty:
                break

        while True:

            try:

                item = self.audio_queue.get_nowait()

                if item is not None:

                    with self._count_lock:

                        self._pending_count -= 1

                self.audio_queue.task_done()

            except queue.Empty:
                break

        with self._count_lock:

            self._pending_count = 0

            self.speech_queued.clear()

        print("[KOKORO] Pipeline flushed.")

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