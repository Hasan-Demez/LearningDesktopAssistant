import queue
import threading

import pyttsx3


class SpeechOutputEngine:
    """Self-contained pyttsx3 TTS engine.

    Signals (main.py reads / controls these):
        speaking — threading.Event — set while speaking
        done     — threading.Event — set after each utterance
    """

    def __init__(
        self,
        voice_index: int = 0,
        rate: int = 200,
        volume: float = 1.0,
    ):

        self.voice_index = voice_index
        self.rate = rate
        self.volume = volume

        self.queue = queue.Queue()

        self.running = True

        # ==================================================
        # SIGNALS (read by main.py)
        # ==================================================

        self.speaking = threading.Event()
        self.done = threading.Event()

        # ==================================================
        # STATE
        # ==================================================

        self.engine = None

        # ==================================================
        # WORKER
        # ==================================================

        self.thread = threading.Thread(
            target=self._worker,
            daemon=True
        )

        self.thread.start()

    # ======================================================
    # ENGINE CREATION
    # ======================================================

    def _create_engine(self):

        engine = pyttsx3.init()

        engine.setProperty(
            "rate",
            self.rate
        )

        engine.setProperty(
            "volume",
            self.volume
        )

        voices = engine.getProperty("voices")

        if not voices:
            raise RuntimeError(
                "No TTS voices were found."
            )

        if (
            self.voice_index < 0
            or self.voice_index >= len(voices)
        ):
            raise ValueError(
                f"Invalid voice_index "
                f"{self.voice_index}. "
                f"Available voices: "
                f"0-{len(voices) - 1}"
            )

        engine.setProperty(
            "voice",
            voices[self.voice_index].id
        )

        return engine

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

        self.done.clear()

        self.queue.put(text)

    # ======================================================
    # WORKER
    # ======================================================

    def _worker(self):

        while self.running:

            text = self.queue.get()

            if text is None:

                self.queue.task_done()

                break

            engine = None

            try:

                self.speaking.set()

                # ------------------------------------------
                # Create TTS engine
                # ------------------------------------------

                engine = self._create_engine()

                self.engine = engine

                # ------------------------------------------
                # Speak
                # ------------------------------------------

                engine.say(text)
                engine.runAndWait()

            except Exception as e:

                print(
                    f"TTS error while saying "
                    f"{text!r}: {e}"
                )

            finally:

                # ------------------------------------------
                # Stop engine
                # ------------------------------------------

                if engine is not None:

                    try:
                        engine.stop()
                    except Exception:
                        pass

                engine = None
                self.engine = None

                self.speaking.clear()

                self.done.set()

                self.queue.task_done()

    # ======================================================
    # STATUS
    # ======================================================

    def is_speaking(self):

        return self.speaking.is_set()

    # ======================================================
    # WAIT
    # ======================================================

    def wait_until_done(self):

        self.queue.join()

    # ======================================================
    # STOP
    # ======================================================

    def stop(self):

        if not self.running:
            return

        self.running = False

        self.queue.put(None)

        if self.engine is not None:

            try:
                self.engine.stop()
            except Exception:
                pass

        self.thread.join(timeout=2)

        self.engine = None