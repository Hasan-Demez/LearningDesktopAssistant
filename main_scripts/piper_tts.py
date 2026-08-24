import queue
import threading

import sounddevice as sd
from piper import PiperVoice


class PiperSpeechOutputEngine:

    def __init__(
        self,
        speech_input,
        model_path: str,
        use_cuda: bool = False,
        volume: float = 1.0,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w_scale: float = 0.8,
    ):
        self.speech_input = speech_input

        self.model_path = model_path
        self.use_cuda = use_cuda

        self.volume = volume
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w_scale = noise_w_scale

        self.queue = queue.Queue()

        self.running = True
        self.speaking = False

        self.speaking_event = threading.Event()

        self.voice = None
        self.stream = None

        self.thread = threading.Thread(
            target=self._worker,
            daemon=True,
        )

        # Load Piper ONCE.
        self._load_voice()

        self.thread.start()

    # ======================================================
    # LOAD PIPER
    # ======================================================

    def _load_voice(self):

        try:

            print(
                f"Loading Piper voice: "
                f"{self.model_path}"
            )

            self.voice = PiperVoice.load(
                self.model_path,
                use_cuda=self.use_cuda,
            )

            print("Piper voice loaded.")

        except Exception as e:

            raise RuntimeError(
                f"Failed to load Piper voice "
                f"'{self.model_path}': {e}"
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

            previous_large_state = None

            try:

                # ------------------------------------------
                # Pause ONLY large Vosk
                # ------------------------------------------

                previous_large_state = (
                    self.speech_input.pause_large()
                )

                self.speaking = True
                self.speaking_event.set()

                # ------------------------------------------
                # Piper synthesis + playback
                # ------------------------------------------

                self._speak(text)

            except Exception as e:

                print(
                    f"Piper TTS error while saying "
                    f"{text!r}: {e}"
                )

            finally:

                # ------------------------------------------
                # Stop audio stream
                # ------------------------------------------

                self._close_stream()

                self.speaking = False
                self.speaking_event.clear()

                # ------------------------------------------
                # Restore ONLY large Vosk
                # ------------------------------------------

                if previous_large_state is not None:

                    try:

                        self.speech_input.resume_large(
                            previous_large_state
                        )

                    except Exception as e:

                        print(
                            "Error resuming large "
                            f"speech input: {e}"
                        )

                self.queue.task_done()

    # ======================================================
    # SPEAK
    # ======================================================

    def _speak(self, text: str):

        stream = None

        first_chunk = True

        try:

            for chunk in self.voice.synthesize(
                text
            ):

                if not self.running:
                    break

                # ------------------------------------------
                # Create audio stream from Piper format
                # ------------------------------------------

                if first_chunk:

                    stream = sd.RawOutputStream(
                        samplerate=chunk.sample_rate,
                        channels=chunk.sample_channels,
                        dtype="int16",
                    )

                    stream.start()

                    self.stream = stream

                    first_chunk = False

                # ------------------------------------------
                # Play Piper audio chunk
                # ------------------------------------------

                stream.write(
                    chunk.audio_int16_bytes
                )

        finally:

            if stream is not None:

                try:
                    stream.stop()
                except Exception:
                    pass

                try:
                    stream.close()
                except Exception:
                    pass

            if self.stream is stream:
                self.stream = None

    # ======================================================
    # CLOSE STREAM
    # ======================================================

    def _close_stream(self):

        if self.stream is None:
            return

        try:
            self.stream.stop()
        except Exception:
            pass

        try:
            self.stream.close()
        except Exception:
            pass

        self.stream = None

    # ======================================================
    # STATUS
    # ======================================================

    def is_speaking(self):

        return self.speaking

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

        # Stop currently playing audio.
        self._close_stream()

        # Wake worker if it is waiting.
        self.queue.put(None)

        self.thread.join(timeout=2)

        self._close_stream()

        self.voice = None