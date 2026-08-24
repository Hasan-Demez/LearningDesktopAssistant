import json
import queue
import threading

import networkx as nx

from llama_cpp import Llama
from stream2sentence import generate_sentences


class DeskLLM:

    def __init__(
        self,
        model_path="models/gguf_models/Qwen3.5-2B-Q4_K_M.gguf",
        n_ctx=1024,
        n_threads=4,
    ):

        print("Loading LLM...")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )

        print("LLM loaded.")

        # ----------------------------------------------
        # Knowledge graph
        # ----------------------------------------------

        self.graph = nx.Graph()

        # ----------------------------------------------
        # Conversation memory
        # ----------------------------------------------

        self.conversation_history = []

        # ----------------------------------------------
        # Mode
        # ----------------------------------------------

        self.mode = "idle"

        self.lock = threading.Lock()

        # ----------------------------------------------
        # LLM request queue + sentence output
        # ----------------------------------------------

        self.request_queue = queue.Queue()

        self.sentence_queue = queue.Queue()

        self.is_generating = threading.Event()

        self.running = True

        self.worker_thread = threading.Thread(
            target=self._worker,
            daemon=True
        )

        self.worker_thread.start()

    # ======================================================
    # MODE
    # ======================================================

    def set_mode(self, mode):

        valid_modes = {
            "idle",
            "chatting",
            "listening",
            "talking",
        }

        if mode not in valid_modes:

            raise ValueError(
                f"Invalid mode: {mode}"
            )

        with self.lock:

            # ------------------------------------------
            # Clear conversation memory when going idle
            # ------------------------------------------

            if mode == "idle":
                self.conversation_history.clear()
                print("Conversation memory cleared.")

            self.mode = mode

    def get_mode(self):

        with self.lock:
            return self.mode

    # ======================================================
    # SUBMIT SPEECH
    # ======================================================

    def submit(self, text):

        if not text:
            return

        text = text.strip()

        if not text:
            return

        with self.lock:
            mode = self.mode

        if mode == "idle":
            return

        self.request_queue.put(
            (mode, text)
        )

    # ======================================================
    # GET SENTENCE
    # ======================================================

    def get_sentence(self):

        try:

            return self.sentence_queue.get_nowait()

        except queue.Empty:

            return None

    # ======================================================
    # WORKER
    # ======================================================

    def _worker(self):

        while self.running:

            try:

                mode, text = self.request_queue.get(
                    timeout=0.1
                )

            except queue.Empty:

                continue

            try:

                # ------------------------------------------
                # Ignore stale requests
                # ------------------------------------------

                with self.lock:
                    current_mode = self.mode

                if mode != current_mode:

                    continue

                # ------------------------------------------
                # Set generating signal
                # ------------------------------------------

                self.is_generating.set()

                # ------------------------------------------
                # Streaming modes (chatting, talking)
                # ------------------------------------------

                if mode == "chatting":

                    messages = self._chat_messages(
                        text
                    )

                    full_response = self._stream_to_queue(
                        messages,
                        temperature=0.7,
                        max_tokens=256,
                    )

                    # ----------------------------------
                    # Store in conversation memory
                    # ----------------------------------

                    self._store_memory(
                        text, full_response
                    )

                elif mode == "talking":

                    messages = self._answer_messages(
                        text
                    )

                    full_response = self._stream_to_queue(
                        messages,
                        temperature=0.3,
                        max_tokens=256,
                    )

                    # ----------------------------------
                    # Store in conversation memory
                    # ----------------------------------

                    self._store_memory(
                        text, full_response
                    )

                # ------------------------------------------
                # Non-streaming mode (listening)
                # ------------------------------------------

                elif mode == "listening":

                    response = self._learn(text)

                    if response:

                        self.sentence_queue.put(
                            response
                        )

                        # ------------------------------
                        # Store in conversation memory
                        # ------------------------------

                        self._store_memory(
                            text, response
                        )

            except Exception as e:

                print(
                    f"LLM worker error: {e}"
                )

            finally:

                self.is_generating.clear()

                self.request_queue.task_done()

    # ======================================================
    # CONVERSATION MEMORY
    # ======================================================

    MAX_HISTORY_TURNS = 10

    def _store_memory(self, user_text, assistant_text):

        with self.lock:

            self.conversation_history.append(
                {
                    "role": "user",
                    "content": user_text,
                }
            )

            self.conversation_history.append(
                {
                    "role": "assistant",
                    "content": assistant_text,
                }
            )

            # ------------------------------------------
            # Trim oldest turns to stay in context
            # ------------------------------------------

            max_messages = (
                self.MAX_HISTORY_TURNS * 2
            )

            if (
                len(self.conversation_history)
                > max_messages
            ):

                self.conversation_history = (
                    self.conversation_history[
                        -max_messages:
                    ]
                )

    def _get_memory_messages(self):

        with self.lock:

            return list(
                self.conversation_history
            )

    # ======================================================
    # CHAT
    # ======================================================

    def _chat_messages(self, user_text):

        system = {
            "role": "system",
            "content": (
                "You are Buddy, a helpful local desk "
                "assistant. Have a natural conversation "
                "with the user. Keep responses concise "
                "because they will be spoken aloud."
            ),
        }

        history = self._get_memory_messages()

        return [system] + history + [
            {
                "role": "user",
                "content": user_text,
            },
        ]

    # ======================================================
    # LEARN
    # ======================================================

    def _learn(self, user_text):

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise knowledge extraction "
                    "system.\n\n"

                    "Extract the most important relationship "
                    "from the user's statement.\n\n"

                    "Return ONLY valid JSON:\n"

                    '{"source": "concept1", '
                    '"target": "concept2", '
                    '"relation": "relationship"}\n\n'

                    "Do not include markdown."
                    "Do not include explanations."
                ),
            },
            {
                "role": "user",
                "content": user_text,
            },
        ]

        response = self._generate(
            messages,
            temperature=0.1,
            max_tokens=150,
        )

        try:

            data = self._extract_json(
                response
            )

            source = str(
                data["source"]
            ).strip()

            target = str(
                data["target"]
            ).strip()

            relation = str(
                data["relation"]
            ).strip()

            self.graph.add_edge(
                source,
                target,
                relationship=relation
            )

            print(
                f"GRAPH: {source} "
                f"--({relation})--> "
                f"{target}"
            )

            return (
                f"I learned that "
                f"{source} {relation} {target}."
            )

        except Exception as e:

            print(
                f"Knowledge extraction failed: {e}"
            )

            self.graph.add_node(
                user_text[:100],
                raw_information=user_text,
            )

            return (
                "I couldn't structure that information, "
                "but I saved it."
            )

    # ======================================================
    # KNOWLEDGE ANSWER
    # ======================================================

    def _answer_messages(self, user_text):

        context = self._get_relevant_context(
            user_text
        )

        system = {
            "role": "system",
            "content": (
                "You are Buddy, a local desk assistant.\n\n"

                "Answer using ONLY the information in "
                "the knowledge base.\n\n"

                "Do not invent facts.\n"

                "If the knowledge base does not contain "
                "the answer, say you don't know.\n\n"

                "Keep the answer concise.\n\n"

                f"KNOWLEDGE BASE:\n{context}"
            ),
        }

        history = self._get_memory_messages()

        return [system] + history + [
            {
                "role": "user",
                "content": user_text,
            },
        ]

    # ======================================================
    # GRAPH SEARCH
    # ======================================================

    def _get_relevant_context(self, query):

        query = query.lower()

        facts = []

        for node in self.graph.nodes:

            node_text = str(node)

            if node_text.lower() in query:

                neighbors = list(
                    self.graph.neighbors(node)
                )

                for neighbor in neighbors:

                    edge = (
                        self.graph.get_edge_data(
                            node,
                            neighbor
                        )
                    )

                    relation = edge.get(
                        "relationship",
                        "is related to"
                    )

                    facts.append(
                        f"- {node} "
                        f"{relation} "
                        f"{neighbor}"
                    )

        if not facts:

            return "No relevant information found."

        return "\n".join(
            dict.fromkeys(facts)
        )

    # ======================================================
    # STREAMING GENERATION
    # ======================================================

    def _generate_streaming(
        self,
        messages,
        temperature=0.7,
        max_tokens=256,
    ):

        stream = self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in stream:

            delta = (
                chunk["choices"][0]
                .get("delta", {})
            )

            content = delta.get("content", "")

            if content:

                yield content

    # ======================================================
    # STREAM TO SENTENCE QUEUE
    # ======================================================

    def _stream_to_queue(
        self,
        messages,
        temperature=0.7,
        max_tokens=256,
    ):

        token_stream = self._generate_streaming(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        all_sentences = []

        for sentence in generate_sentences(
            token_stream,
            tokenizer="rule-based",
            language="en",
            minimum_sentence_length=20,
            auto_context=True,
            quick_yield_single_sentence_fragment=True,
            cleanup_text_links=True,
        ):

            sentence = sentence.strip()

            if sentence:

                all_sentences.append(sentence)

                self.sentence_queue.put(
                    sentence
                )

        full_response = " ".join(
            all_sentences
        )

        return full_response

    # ======================================================
    # GENERATION (non-streaming, used by _learn)
    # ======================================================

    def _generate(
        self,
        messages,
        temperature=0.7,
        max_tokens=256,
    ):

        response = (
            self.llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )

        return (
            response["choices"][0]
            ["message"]["content"]
            .strip()
        )

    # ======================================================
    # JSON
    # ======================================================

    @staticmethod
    def _extract_json(text):

        try:

            return json.loads(text)

        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:

            raise ValueError(
                "No JSON object found."
            )

        return json.loads(
            text[start:end + 1]
        )

    # ======================================================
    # SHUTDOWN
    # ======================================================

    def stop(self):

        self.running = False

        with self.lock:
            self.conversation_history.clear()

        self.worker_thread.join(
            timeout=2
        )