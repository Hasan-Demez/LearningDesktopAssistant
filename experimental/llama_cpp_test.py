import lark
from llama_cpp import Llama

# Define the exact path where you saved your downloaded model file
SMOLLM_MODEL_PATH = "models/gguf_models/SmolLM2-135M-Instruct-Q4_K_M.gguf"
QWEN_MODEL_PATH = "models/gguf_models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
print(f"Loading local GGUF model from: {SMOLLM_MODEL_PATH}")

# Initialize directly from your local file system (No internet required)
llm = Llama(
    model_path=SMOLLM_MODEL_PATH,
    n_ctx=512,        # Small context window saves RAM
    n_threads=4,      
    verbose=False     
)

# Your parsing pipeline continues here...
input = ""

messages = [
    {
        "role": "system",
        "content": (
            "Convert mathematical expressions into natural spoken English. "
            "Return only the expression. "
            "Do not explain anything."
        )
    },
    {
        "role": "user",
        "content": f"{input}"
    }
]

response = llm.create_chat_completion(messages=messages, temperature=0.0)
natural_text = response["choices"][0]["message"]["content"].strip()

print(natural_text)
