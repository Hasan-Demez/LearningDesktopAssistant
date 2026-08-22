## Goal

So the goal of this project is not a home assistant but a desktop assistant that can learn and repeat back to you what you taught it along with potential future upgrades like becoming an agent.

## Tech Stack

So the current tech stack for english only is:

Audio input: sounddevice

speech to text: Vosk small for keywords and vosk large for learning

Text processing: lark

Knowledge graph: networkx

Visualize knowledge graph: matplotlib

LLM Runtime: llama-cpp-python: llama-cpp-python   
\--extra-index-url [https://abetlen.github.io/llama-cpp-python/whl/cpu](https://abetlen.github.io/llama-cpp-python/whl/cpu)

LLM models: SmolLM2-135M-Instruct for lark IO conversion, Qwen2.5-0.5B-Instruct for potential brain

Text to Speech: for a robotic and fast speech it has to be pyttsx3

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
