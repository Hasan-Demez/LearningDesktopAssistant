# Project

## Goal

So the goal of this project is not a home assistant but a desktop assistant that can learn and repeat back to you what you taught it along with potential future upgrades like becoming an agent.

## Tech Stack

So the current tech stack for english only is:

Audio input: sounddevice

speech to text: Vosk small for keywords, Faster-Whisper for llm handling

Voice Activity Detection: TEN VAD

LLM Runtime: llama-cpp-python: llama-cpp-python   
\--extra-index-url [https://abetlen.github.io/llama-cpp-python/whl/cpu](https://abetlen.github.io/llama-cpp-python/whl/cpu)

LLM models: Qwen3.5-2B-Instruct for brain

Text to Speech: for a robotic and fast speech pyttsx3, for better sounding llm kokoro-onnx

Utility: stream2sentence for llm stream to kokoro sentence.
## Theory

LLMs use a deep neural network. This has many problems but the three i will focus on is that DNNs are a black box, DNNs do not know anything, DNNs cannot learn easily without fine tuning much of the model and compiling it. While a similar argument could be made for the human brain in that the human brain is a black box, it can know things and learn things. But the human brain, in its simple form, is a bunch of neurons(nodes) with plasticity(weights). We also learn things like machine learning according to behaviorists. We learn through reinforcement and punishment.

I would like to propose an alternative perspective to the common biological behaviorist narrative. A more mentalist perspective involves a concept I would like to call a concept map. The idea is that we have the ability to know distinct concepts like nodes each with relationships to other concepts like edges. This is also based on the value of language and words from an existential perspective as it was what got humans to where we are.

The benefits of this concept map turned knowledge graph is that the language model would no longer be a blackbox, would actually know things, and would be able to learn new information.

The goal of this theoretical model is not to recreate a human with its flaws but make an ideal human mind, never to forget and to instantly learn and never hallucinate.

Not to say that the two cannot be combined. You might need to add weights in order for it to make decisions. But for this particular project, it only needs to prove it can learn and know.

## Importance

The importance of this project is to be a minimum viable product to prove that a knowledge base can learn and know concepts. This can be used to make more elaborate projects in the future.

## Future

I would like this to be modular and be able to do more in the future but as a desk assistant this is enough. Eventually I want to build a vr game engine based on this that is fully procedural. It will have physics and every character will have its own brain with a certain level of access to a central knowledge base.

An example of this brain would look like:

Character Brain=\> Goal \-\> Milestone \-\> Sense \-\> Short Term memory \-\> Think \-\> Decide-\> Behave \-\> Integrate \-\> Coordination

I hope to model decision making of this assistant in the future based on this. This Character Brain also needs to coordinate between muscle motors for physics so that is why its there but it can be converted to a desk assistant.

A Goal is the main mission of the model. The milestone is the closest step of a goal split into smaller steps. Sense only recognizes things relevant to the milestone and records it to short term memory. The think uses all the information available to it to process and provide it to the decision maker. The decision makes an abstract decision and the behavior puts it to practice, the integrate finds what muscles need activation and coordinate between different muscles. This is not exactly an accurate demonstration but I would use this for decision making in the future is all I would like to say.

Another future plan is to make everything not use neural networks but rather use concepts including speech to text. Or as I said before to combine the two.

## Potential Hardware

### Computer Mini (Recommended)
Lenovo ThinkCentre M710q Tiny PC 16b
Dell Optiplex 3040 mico pc 16gb
Dell wyse 5070 upgrade to 16gb
Basically 16gb or 8gb could also work too for now.

### SpeakerPhone
EMEET OfficeCore M1A Speakerphone

### Headset
Corsair HS55 WIRELESS CORE Gaming Headset.



## Change Log
Here is the changelog so far

### first commit
First commmit as well as only having the vosk and pyttsx3 without the brain.

### first main release
First release. Decided to make the project heavier after testing llm with first piper then didnt like the voice so switched to kokoro. Similarly did not like vosk large detection so switched to whisperx. Might as well go all the way. In hindsight i should have commited and pushed the llm with only vosk and pttsx3 then vosk and piper then whisperx then kokoro but this is how it is i suppose.
Added setup_models.py for auto downloading models.
Uses Qwen3.5 2b as brain might change to 4b but found no reason as its good enough.
Uses Kokoro 82m which is really good and no reason to change.
Uses Faster Whisper small instead of base and its accurate enough now.

Currently only chatting mode is tested.
This means that no knowledge graph or concept map. A basic local chat bot wihtout memory.

### A better first release

Replaced WhisperX with Faster-Whisper + TEN VAD for better performance.
Removed knowledge graph and learning modes (listening/talking).
Simplified to only idle and chatting modes.
Added WhisperController for pause/continue/stop commands during chatting.
VoskController handles wake/sleep/chat mode transitions.
TEN VAD is always active during chatting mode, never paused.
Beep plays only when it's the user's turn to speak.




## First usage
run setup_models then run main. It might still need internet usage for the first run. It should work.
You wait until the beep to speak. The first beep when it says "turned on" say "wake up buddy" to put it in idle mode. The second beep after hearing "I'm Awake" say "start chatting buddy" to put it in chat mode. When you hear the beep you can speak. To return to idle mode(not chatting) say "stop chatting buddy" at any time when in chat mode. To make it go to sleep say "sleep now" at any time.

Idle mode is awake but waiting command. Chat mode is chatting with the llm.
