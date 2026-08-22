import queue
import threading

import pyttsx3


class SpeechOutputEngine:

    def __init__(
        self,
        speech_input,
        voice_index: int = 0,
        rate: int = 200,
        volume: float = 1.0,
    ):
        self.speech_input = speech_input

        self.voice_index = voice_index
        self.rate = rate
        self.volume = volume

        self.queue = queue.Queue()

        self.running = True
        self.speaking = False

        self.speaking_event = threading.Event()

        # The engine is intentionally NOT created here.
        #
        # pyttsx3 is created inside the worker thread instead.
        self.engine = None

        self.thread = threading.Thread(
            target=self._worker,
            daemon=True
        )

        self.thread.start()

    # ======================================================
    # ENGINE CREATION
    # ======================================================

    def _create_engine(self):
        """
        Create and configure a completely new pyttsx3
        engine.

        This is called from the worker thread immediately
        before speaking.
        """

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
                f"Invalid voice_index {self.voice_index}. "
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
        """
        Add text to the speech queue.

        This does NOT block.
        """

        if not self.running:
            return

        if not text:
            return

        text = str(text).strip()

        if not text:
            return

        self.queue.put(text)

    # ======================================================
    # TTS WORKER
    # ======================================================

    def _worker(self):

        while self.running:

            text = self.queue.get()

            # Shutdown signal
            if text is None:
                self.queue.task_done()
                break

            previous_state = None
            engine = None

            try:

                # ------------------------------------------
                # Pause Vosk
                # ------------------------------------------

                previous_state = self.speech_input.pause()

                self.speaking = True
                self.speaking_event.set()

                # ------------------------------------------
                # Create a NEW engine
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

                # ------------------------------------------
                # Delete engine
                # ------------------------------------------

                engine = None
                self.engine = None

                # ------------------------------------------
                # TTS is finished
                # ------------------------------------------

                self.speaking = False
                self.speaking_event.clear()

                # ------------------------------------------
                # Restore Vosk
                # ------------------------------------------

                if previous_state is not None:

                    try:
                        self.speech_input.resume(
                            previous_state
                        )
                    except Exception as e:
                        print(
                            f"Error resuming speech "
                            f"input: {e}"
                        )

                self.queue.task_done()

    # ======================================================
    # STATUS
    # ======================================================

    def is_speaking(self):
        """
        Returns True if the assistant is currently speaking.
        """

        return self.speaking

    # ======================================================
    # WAIT
    # ======================================================

    def wait_until_done(self):
        """
        Wait until everything currently in the queue
        has finished speaking.
        """

        self.queue.join()

    # ======================================================
    # STOP
    # ======================================================

    def stop(self):

        if not self.running:
            return

        self.running = False

        # Wake the worker if it is waiting on the queue
        self.queue.put(None)

        # Stop the currently active engine
        if self.engine is not None:

            try:
                self.engine.stop()
            except Exception:
                pass

        # Wait for worker to exit
        self.thread.join(timeout=2)

        self.engine = None
