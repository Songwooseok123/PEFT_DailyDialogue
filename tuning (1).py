from peft import get_peft_config, get_peft_model, LoraConfig, PrefixTuningConfig, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
from transformers import BitsAndBytesConfig,LlamaTokenizer,LlamaForCausalLM,GPT2Tokenizer, GPT2LMHeadModel
import torch

from peft import prepare_model_for_kbit_training

def get_model_architecture(tuning_name,model_name,model):
    if tuning_name == "full_ft" : 
        model = model
    else:     
        if tuning_name == "lora" : 
            peft_config = LoraConfig(
                r=64, lora_alpha=128, lora_dropout=0.0, target_modules=["embed_tokens", "lm_head", "q_proj", "v_proj"]
            )
        elif tuning_name == "prefix" :     
            peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30)
        elif tuning_name == "prompt" :     
            peft_config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=20,
                prompt_tuning_init=PromptTuningInit.TEXT,
                prompt_tuning_init_text="Create a sentence that will continue in the next conversation according to the given attributes.", 
                tokenizer_name_or_path="meta-llama/Llama-2-7b-hf"
            )
        model = get_peft_model(model, peft_config)
    if model_name == "llama":
        if tuning_name == "prefix" :
            model = model
        else:
            model = prepare_model_for_kbit_training(model,
                                                    #use_gradient_checkpointing=True,
                                                    gradient_checkpointing_kwargs = {"use_reentrant": True}
                                                   )
            model.config.use_cache = False 
    return model




def get_model(model_name,device):
    if model_name == "llama":
        model_path ="meta-llama/Llama-2-7b-hf"
        nf4_config = BitsAndBytesConfig(
           load_in_4bit=True,
           bnb_4bit_quant_type="nf4",
           bnb_4bit_use_double_quant=True,
           bnb_4bit_compute_dtype=torch.bfloat16
        )
        model =LlamaForCausalLM.from_pretrained(model_path,
                                                #load_in_8bit=True, #  7.7GB로
                                                quantization_config =nf4_config, #  4.4GB로 
                                                device_map="auto" # gpu 꽉차면 cpu로 올려줌 
                                               )
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
    
    else  :
        
        tokenizer = GPT2Tokenizer.from_pretrained(f"microsoft/{model_name}")
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(f"microsoft/{model_name}").to(device)
    

    return tokenizer, model