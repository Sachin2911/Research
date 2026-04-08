import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

model_id = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

#4-bit quantisation to save vram
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,        
    torch_dtype=torch.float16,
    device_map="auto"        
)