from transformers import GPT2LMHeadModel
from peft import LoraConfig, get_peft_model

def apply_lora(model: GPT2LMHeadModel, rank: int = 8):
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=32,
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM'
    )
    model = get_peft_model(model, lora_config)
    return model
