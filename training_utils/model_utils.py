import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from training_utils.configurations import GRPOConfig, SFTConfig
from unsloth import FastLanguageModel


def get_tokenizer(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    return tokenizer


def get_model(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig) -> AutoModelForCausalLM:
    """Get the model"""
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs,
        )
    return model



    
    # Extract unsloth-specific parameters
    # max_seq_length = getattr(model_args, 'max_seq_length', 2048)
    # lora_rank = getattr(model_args, 'lora_r', 16)
    # load_in_4bit = getattr(model_args, 'load_in_4bit', False)
    
    # # Load model and tokenizer with unsloth
    # model, _ = FastLanguageModel.from_pretrained(
    #     model_name=model_args.model_name_or_path,
    #     max_seq_length=max_seq_length,
    #     dtype=model_args.torch_dtype,
    #     load_in_4bit=load_in_4bit,
    #     trust_remote_code=model_args.trust_remote_code,
    #     # Unsloth optimizations
    #     fast_inference=True,
    #     max_lora_rank=lora_rank,
    #     gpu_memory_utilization=0.9,
    # )
    
    # # Setup LoRA with unsloth
    # target_modules = getattr(model_args, 'lora_target_modules', [
    #     "q_proj", "k_proj", "v_proj", "o_proj",
    #     "gate_proj", "up_proj", "down_proj"
    # ]) 
       
    # model = FastLanguageModel.get_peft_model(
    #     model,
    #     r=lora_rank,
    #     target_modules=target_modules,
    #     lora_alpha=getattr(model_args, 'lora_alpha', lora_rank * 2),
    #     lora_dropout=getattr(model_args, 'lora_dropout', 0.05),
    #     bias="none",
    #     use_gradient_checkpointing="unsloth",
    #     random_state=getattr(training_args, 'seed', 42),
    #     use_rslora=False,
    #     loftq_config=None,
    # )
    
    # return model