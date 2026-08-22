from main_scripts.stt import SpeechInputEngine
from main_scripts.tts import SpeechOutputEngine

assistant_name = "buddy"

sleep_keywords = [
    "turn on",
    assistant_name,
    "[unk]"
]

command_keywords = [
    "start listening",
    "stop listening",
]

awake_keywords = [
    "turn off",
    *command_keywords,
    assistant_name,
    "[unk]"
]

speech_input = SpeechInputEngine(sleep_keywords)

speech_output = SpeechOutputEngine(
    speech_input=speech_input
)


speech_input.start()
speech_input.set_small_keywords(sleep_keywords)

# Small model is always active

speech_input.small_on()


try:

    while True:

        # ==============================================
        # SMALL MODEL
        # ==============================================

        small_text = speech_input.get_small()

        if small_text:

            print("COMMAND:", small_text)

            # ------------------------------------------
            # WAKE UP
            # ------------------------------------------

            if (
                "turn on" in small_text
                and assistant_name in small_text
            ):

                speech_input.set_small_keywords(
                    awake_keywords
                )

                speech_output.talk(
                    "I'm awake."
                )

            # ------------------------------------------
            # GO TO SLEEP
            # ------------------------------------------

            elif (
                "turn off" in small_text
                and assistant_name in small_text
            ):

                speech_input.large_off()

                speech_input.set_small_keywords(
                    sleep_keywords
                )

                speech_output.talk(
                    "Going to sleep."
                )

            # ------------------------------------------
            # START LARGE LISTENING
            # ------------------------------------------

            elif ("start listening" in small_text
                and assistant_name in small_text
            ):

                speech_input.large_on()

                speech_output.talk(
                    "I'm listening."
                )

            # ------------------------------------------
            # STOP LARGE LISTENING
            # ------------------------------------------

            elif ("stop listening" in small_text
                and assistant_name in small_text
            ):

                speech_input.large_off()

                speech_output.talk(
                    "I stopped listening."
                )


        # ==============================================
        # LARGE MODEL
        # ==============================================

        large_text = speech_input.get_large()

        if large_text:

            print("SPEECH:", large_text)

            # response = process(large_text)
            #
            # speech_output.talk(response)


except KeyboardInterrupt:

    pass


finally:

    speech_input.stop()
    speech_output.stop()
