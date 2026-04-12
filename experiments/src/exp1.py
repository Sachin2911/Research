#Libraries
import os
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from transformers import AutoModelForCausalLM, AutoTokenizer

class LoRa(nn.Module): 
    """
    Instead of updating W directly, we learn a small additive update: W' = W + (B @ A) * (alpha / rank)
    Note B is initialized with zeros — so ΔW = BA = 0 at start (model unchanged)
    """
    def __init__(self, feat_in, feat_out, rank=16, alpha=32, device="cpu", dtype=torch.bfloat16):
        super().__init__() #Inherits from nn.Module
        self.lora_A = nn.Parameter(torch.zeros((rank, feat_out), device=device, dtype=dtype)) #Tells pytorch this is a learnable param not a normal tensor
        nn.init.normal_(self.lora_A, mean=0, std=1) #Makes every ele Gaussian
        
        #The second matrix
        self.lora_B = nn.Parameter(torch.zeros((feat_in, rank), device=device, dtype=dtype))
        
        self.scale = alpha/rank #Controls how much influence the adapter has, check the forward
        self.enabled = True #Whether the adapter is on or off
        
    def forward(self, original_weights):
        if self.enabled:
            return original_weights + (self.lora_B @ self.lora_A) * self.scale
        return original_weights

class Model:
    """
    Generic model class to interact with the llm
    """
    
    def __init__(
        self,
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        lora_rank=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
    ):
        """
        Sets up the model and tokenizer
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_id) #Creates the tokenizer for the model
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id #Set the padding token explicitly
        
        for param in self.model.parameters():
            param.requires_grad = False #Freezing all the base model's params
            
        #Attaching the lora adapters
        self.lora_layers = []
        self.attach_lora(lora_rank, lora_alpha, target_modules)
        
        self.lora_param_count = sum(
            lora.lora_A.numel() + lora.lora_B.numel()
            for lora in self.lora_layers
        )
        print(f"LoRA params: {self.lora_param_count:,}")
        
    def attach_lora(self, rank, alpha, target_modules):
        """
        Here walk through all layers and attach LoRA parametrization to linear layers matching target_modules names.
        """
        for name, module in self.model.named_modules():
            module_name = name.split(".")[-1] #We just getting the last part so we can match with the module names we gave
            
            if module_name in target_modules and isinstance(module, nn.Linear):
                feat_in, feat_out = module.weight.shape
                
                lora = LoRa(feat_in, feat_out, rank=rank, alpha=alpha, device=module.weight.device, dtype=module.weight.dtype)
                parametrize.register_parametrization(module, "weight", lora)
                self.lora_layers.append(lora)
                # print(f"  LoRA attached: {name} ({feat_in}x{feat_out})")
        
        
    def get_model_struct(self, filepath="logs/model_structure.txt"):
        """Writes the model structure to a file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    f.write(f"{name}: {module.weight.shape}\n")
                else:
                    f.write(f"{name}: {type(module).__name__}\n")
        
        print(f"Model structure written to {filepath}")
        
    def gen(self, prompt, max_new_tokens=128):
        """
        Uses the model to generate new text given some input prompt
        """
        messages = [{"role": "user", "content": prompt}]
        """
        Formats the messages into Llama 3.2's expected token structure:
        <|begin_of_text|><|start_header_id|>user<|end_header_id|> {content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        ).to(self.model.device) 
        
        with torch.no_grad():
            #The do sample is necessary coz we don't want a multinomial pick just straight
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.eos_token_id,)
            return self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

llama = Model()
print(llama.gen("Hi there!"))
# llama.get_model_struct()\