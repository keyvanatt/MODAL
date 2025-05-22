import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType

class QLoRAEncoder(nn.Module):
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", lora_r=8, lora_alpha=16, lora_dropout=0.05, pool=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            use_auth_token=True
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],  # à adapter selon le modèle
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.model = get_peft_model(base_model, lora_config)
        self.dim = self.model.config.hidden_size
        self.pool = pool

    def forward(self, texts):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
        outputs = self.model.model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        if self.pool:
            pooled = last_hidden_state.mean(dim=1)
            return pooled
        else:
            return last_hidden_state