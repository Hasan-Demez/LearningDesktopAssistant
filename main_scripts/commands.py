class AssistantController:

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

            case "start_listening":

                self.start_listening()

            case "stop_listening":

                self.stop_listening()

            case "start_talking":

                self.start_talking()

            case "stop_talking":

                self.stop_talking()

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

        # ----------------------------------------------
        # Learning controls
        # ----------------------------------------------

        if (
            "start listening" in text
            and self.assistant_name in text
        ):
            return "start_listening"

        if (
            "stop listening" in text
            and self.assistant_name in text
        ):
            return "stop_listening"

        # ----------------------------------------------
        # Knowledge controls
        # ----------------------------------------------

        if (
            "start talking" in text
            and self.assistant_name in text
        ):
            return "start_talking"

        if (
            "stop talking" in text
            and self.assistant_name in text
        ):
            return "stop_talking"

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
    # START LISTENING
    # ======================================================

    def start_listening(self):

        self.awake = True

        self.llm.set_mode(
            "listening"
        )

        self.speech_input.set_small_keywords(
            self.learning_control_keywords()
        )

        self._emit_mode_change("listening")

        self.speech_output.talk(
            "I'm listening."
        )

        print("MODE: LISTENING")

    # ======================================================
    # STOP LISTENING
    # ======================================================

    def stop_listening(self):

        self.llm.set_mode(
            "idle"
        )

        self.speech_input.set_small_keywords(
            self.awake_keywords()
        )

        self._emit_mode_change("idle")

        self.speech_output.talk(
            "I stopped listening."
        )

        print("MODE: IDLE")

    # ======================================================
    # START TALKING
    # ======================================================

    def start_talking(self):

        self.awake = True

        self.llm.set_mode(
            "talking"
        )

        self.speech_input.set_small_keywords(
            self.knowledge_control_keywords()
        )

        self._emit_mode_change("talking")

        self.speech_output.talk(
            "I'm ready."
        )

        print("MODE: TALKING")

    # ======================================================
    # STOP TALKING
    # ======================================================

    def stop_talking(self):

        self.llm.set_mode(
            "idle"
        )

        self.speech_input.set_small_keywords(
            self.awake_keywords()
        )

        self._emit_mode_change("idle")

        self.speech_output.talk(
            "I stopped talking."
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
            "start listening",
            "start talking",

            self.assistant_name,

            "[unk]",
        ]

    def chat_control_keywords(self):

        return [
            "stop chatting",
            "sleep now",
            self.assistant_name,
        ]

    def learning_control_keywords(self):

        return [
            "stop listening",
            "sleep now",
            self.assistant_name,
        ]

    def knowledge_control_keywords(self):

        return [
            "stop talking",
            "sleep now",
            self.assistant_name,
        ]