class VoskController:
    """Vosk-based controller for wake/sleep/chat mode transitions.

    Handles commands spoken via Vosk (small model):
        - "wake up buddy" → turn on
        - "sleep now buddy" → turn off
        - "start chatting buddy" → enter chatting mode
        - "stop chatting buddy" → exit chatting mode
    """

    def __init__(
        self,
        speech_input,
        speech_output,
        llm,
        assistant_name="buddy",
    ):

        self.speech_input = speech_input
        self.speech_output = speech_output
        self.llm = llm
        self.assistant_name = assistant_name

        self.awake = False

        # ==================================================
        # SIGNALS (emitted by controller, consumed by main.py)
        # ==================================================

        self._on_mode_change = None
        self._on_state_change = None

    # ======================================================
    # SIGNAL REGISTRATION
    # ======================================================

    def on_mode_change(self, callback):
        """Register a callback for mode changes.

        callback(mode: str, awake: bool)
        """
        self._on_mode_change = callback

    def on_state_change(self, callback):
        """Register a callback for state changes.

        callback(awake: bool)
        """
        self._on_state_change = callback

    def _emit_mode_change(self, mode):
        """Emit mode change signal."""
        if self._on_mode_change:
            self._on_mode_change(mode, self.awake)

    def _emit_state_change(self):
        """Emit state change signal."""
        if self._on_state_change:
            self._on_state_change(self.awake)

    # ======================================================
    # COMMAND ENTRY
    # ======================================================

    def handle(self, text):

        text = text.lower().strip()

        if not text:
            return

        command = self._identify_command(
            text
        )

        if command is None:
            return

        match command:

            case "turn_on":

                self.turn_on()

            case "turn_off":

                self.turn_off()

            case "start_chatting":

                self.start_chatting()

            case "stop_chatting":

                self.stop_chatting()

    # ======================================================
    # COMMAND IDENTIFICATION
    # ======================================================

    def _identify_command(self, text):

        # ----------------------------------------------
        # Wake / sleep controls
        # ----------------------------------------------

        if (
            "wake up" in text
            and self.assistant_name in text
        ):
            return "turn_on"

        if (
            "sleep now" in text
            and self.assistant_name in text
        ):
            return "turn_off"

        # ----------------------------------------------
        # Chat controls
        # ----------------------------------------------

        if (
            "start chatting" in text
            and self.assistant_name in text
        ):
            return "start_chatting"

        if (
            "stop chatting" in text
            and self.assistant_name in text
        ):
            return "stop_chatting"

        return None

    # ======================================================
    # TURN ON
    # ======================================================

    def turn_on(self):

        self.awake = True

        self.llm.set_mode(
            "idle"
        )

        self.speech_input.set_small_keywords(
            self.awake_keywords()
        )

        self._emit_state_change()
        self._emit_mode_change("idle")

        self.speech_output.talk(
            "I'm awake."
        )

        print("STATE: AWAKE")

    # ======================================================
    # TURN OFF
    # ======================================================

    def turn_off(self):

        self.awake = False

        self.llm.set_mode(
            "idle"
        )

        self.speech_input.set_small_keywords(
            self.sleep_keywords()
        )

        self._emit_state_change()
        self._emit_mode_change("idle")

        self.speech_output.talk(
            "Going to sleep."
        )

        print("STATE: SLEEP")

    # ======================================================
    # START CHAT
    # ======================================================

    def start_chatting(self):

        self.awake = True

        self.llm.set_mode(
            "chatting"
        )

        self.speech_input.set_small_keywords(
            self.chat_control_keywords()
        )

        self._emit_mode_change("chatting")

        self.speech_output.talk(
            "Chat mode."
        )

        print("MODE: CHATTING")

    # ======================================================
    # STOP CHAT
    # ======================================================

    def stop_chatting(self):

        self.llm.set_mode(
            "idle"
        )

        self.speech_input.set_small_keywords(
            self.awake_keywords()
        )

        self._emit_mode_change("idle")

        self.speech_output.talk(
            "Chat mode stopped."
        )

        print("MODE: IDLE")

    # ======================================================
    # KEYWORDS
    # ======================================================

    def sleep_keywords(self):

        return [
            "wake up",
            self.assistant_name,
            "[unk]",
        ]

    def awake_keywords(self):

        return [
            "sleep now",

            "start chatting",

            self.assistant_name,

            "[unk]",
        ]

    def chat_control_keywords(self):

        return [
            "stop chatting",
            "sleep now",
            self.assistant_name,
        ]


class WhisperController:
    """Whisper-based controller for commands during chatting mode.

    Handles commands spoken via Faster-Whisper (full transcription):
        - "pause buddy" → pause Kokoro
        - "continue buddy" → unpause Kokoro
        - "stop buddy" → back to idle mode
    """

    def __init__(
        self,
        llm,
        llm_tts,
        speech_output,
        assistant_name="buddy",
    ):

        self.llm = llm
        self.llm_tts = llm_tts
        self.speech_output = speech_output
        self.assistant_name = assistant_name

        self.paused = False

        # ==================================================
        # SIGNALS (emitted by controller, consumed by main.py)
        # ==================================================

        self._on_mode_change = None
        self._on_pause_change = None

    # ======================================================
    # SIGNAL REGISTRATION
    # ======================================================

    def on_mode_change(self, callback):
        """Register a callback for mode changes.

        callback(mode: str, awake: bool)
        """
        self._on_mode_change = callback

    def on_pause_change(self, callback):
        """Register a callback for pause state changes.

        callback(paused: bool)
        """
        self._on_pause_change = callback

    def _emit_mode_change(self, mode):
        """Emit mode change signal."""
        if self._on_mode_change:
            self._on_mode_change(mode, True)

    def _emit_pause_change(self):
        """Emit pause state change signal."""
        if self._on_pause_change:
            self._on_pause_change(self.paused)

    # ======================================================
    # COMMAND ENTRY
    # ======================================================

    def handle(self, text):
        """Handle transcribed text from Whisper.

        Returns True if a command was handled, False otherwise.
        """

        text = text.lower().strip()

        if not text:
            return False

        command = self._identify_command(
            text
        )

        if command is None:
            return False

        match command:

            case "pause":

                self.pause()

            case "continue":

                self.continue_speaking()

            case "stop":

                self.stop()

        return True

    # ======================================================
    # COMMAND IDENTIFICATION
    # ======================================================

    def _identify_command(self, text):

        # ----------------------------------------------
        # Pause
        # ----------------------------------------------

        if (
            "pause" in text
            and self.assistant_name in text
        ):
            return "pause"

        # ----------------------------------------------
        # Continue
        # ----------------------------------------------

        if (
            "continue" in text
            and self.assistant_name in text
        ):
            return "continue"

        # ----------------------------------------------
        # Stop
        # ----------------------------------------------

        if (
            "stop" in text
            and self.assistant_name in text
        ):
            return "stop"

        return None

    # ======================================================
    # PAUSE
    # ======================================================

    def pause(self):

        self.paused = True

        self._emit_pause_change()

        self.speech_output.talk(
            "Paused."
        )

        print("WHISPER: PAUSED")

    # ======================================================
    # CONTINUE
    # ======================================================

    def continue_speaking(self):

        self.paused = False

        self._emit_pause_change()

        self.speech_output.talk(
            "Continuing."
        )

        print("WHISPER: CONTINUED")

    # ======================================================
    # STOP
    # ======================================================

    def stop(self):

        self.paused = False

        self.llm.set_mode(
            "idle"
        )

        # Drop everything Kokoro still has queued
        # or stashed from a previous pause.
        self.llm_tts.flush()

        self._emit_pause_change()
        self._emit_mode_change("idle")

        self.speech_output.talk(
            "Chat mode stopped."
        )

        print("WHISPER: STOPPED → IDLE")