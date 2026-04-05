import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-3B"

def load_model():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"Model loaded. Parameters: {model.num_parameters():,}")
    return model, tokenizer

def run_inference(model, tokenizer, prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Hello, I am")
    args = parser.parse_args()

    model, tokenizer = load_model()
    response = run_inference(model, tokenizer, args.prompt)
    print(f"\n--- Response ---\n{response}")