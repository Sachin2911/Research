from transformers import pipeline
import torch

pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-14B-Instruct",
    dtype=torch.float16,
    device_map="auto"
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

result = pipe(
    messages,
    max_new_tokens=1000,
    pad_token_id=pipe.tokenizer.eos_token_id,
)
print(result[0]["generated_text"][-1]["content"])