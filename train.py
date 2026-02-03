from unsloth import FastLanguageModel
import os
import sys
import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOTrainer, TrlParser, ModelConfig
from training_utils.configurations import GRPOConfig, GRPOScriptArguments
import logging
from training_utils.reward import custom_reward_function, mixed_objective_reward_function
from training_utils.train_utils import prepare_dataset
from training_utils.callbacks import get_callbacks

from training_utils.wandb_logging import init_wandb_training
import inspect

logger = logging.getLogger(__name__)

def get_model_with_unsloth(model_args: ModelConfig, training_args: GRPOConfig):
    """Load model and using unsloth for optimized training"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_args.model_name_or_path,
        max_seq_length = training_args.max_prompt_length + training_args.max_completion_length,
        load_in_4bit = False, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = model_args.lora_r,
        gpu_memory_utilization = 0.8, # Reduce if out of memory
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = model_args.lora_r,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = model_args.lora_r * 2,
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )
    return model, tokenizer

def reward_function_wrapper(completions, **kwargs):
    frame = inspect.currentframe()
    try:
        is_eval = False
        current_step = 0
        while frame:
            if 'evaluate' in frame.f_code.co_name or 'eval' in frame.f_code.co_name:
                is_eval = True
                if 'self' in frame.f_locals and hasattr(frame.f_locals['self'], 'state'):
                    current_step = getattr(frame.f_locals['self'].state, 'global_step', 0)
                break
            frame = frame.f_back
    finally:
        del frame
    
    kwargs['is_eval'] = is_eval
    kwargs['current_step'] = current_step
    return custom_reward_function(completions, **kwargs)

def mixed_objective_reward_function_wrapper(completions, **kwargs):
    frame = inspect.currentframe()
    try:
        is_eval = False
        current_step = 0
        while frame:
            if 'evaluate' in frame.f_code.co_name or 'eval' in frame.f_code.co_name:
                is_eval = True
                if 'self' in frame.f_locals and hasattr(frame.f_locals['self'], 'state'):
                    current_step = getattr(frame.f_locals['self'].state, 'global_step', 0)
                break
            frame = frame.f_back
    finally:
        del frame
    
    kwargs['is_eval'] = is_eval
    kwargs['current_step'] = current_step
    return mixed_objective_reward_function(completions, **kwargs)

def main(script_args, training_args, model_args):
        
    set_seed(training_args.seed)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")


    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ##############
    # Load model #
    ##############
    logger.info("*** Loading model and tokenizer ***")
    model, tokenizer = get_model_with_unsloth(model_args, training_args)
    logger.info("Model loaded successfully with unsloth optimizations")
    dataset = prepare_dataset(script_args, training_args.output_dir)
    # Shuffle and split dataset into train and eval
    dataset = dataset.shuffle(seed=42)
    
    # Split dataset (e.g., 80% train, 20% eval)
    test_ratio = 0.2
    # test size <= 30
    test_ratio = min(test_ratio, 30 / len(dataset))
    train_test_split = dataset.train_test_split(
        test_size=test_ratio,
        seed=42
    )
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    
    #############################
    # Initialize the GRPO trainer
    #############################
    # shuffle then split dataset into train and eval
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_function_wrapper if not script_args.use_mixed_objective else mixed_objective_reward_function_wrapper,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer
    )
    
    # original_compute_loss = trainer.compute_loss
    
    # def patched_compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
    #     print(f"\n[COMPUTE_LOSS DEBUG] Called with inputs type: {type(inputs)}")
    #     if isinstance(inputs, dict):
    #         for key, value in inputs.items():
    #             if hasattr(value, 'shape'):
    #                 print(f"[COMPUTE_LOSS DEBUG] {key} shape: {value.shape}")
    #             elif isinstance(value, (list, tuple)):
    #                 print(f"[COMPUTE_LOSS DEBUG] {key} length: {len(value)}")
        
    #     return original_compute_loss(model, inputs, return_outputs, num_items_in_batch)
    
    # trainer.compute_loss = patched_compute_loss
    if not hasattr(trainer, 'current_gradient_accumulation_steps'):
        trainer.current_gradient_accumulation_steps = 0

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(trainer.train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    # Align the model's generation config with the tokenizer's eos token
    # to avoid unbounded generation in the transformers `pipeline()` function
    # trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    # trainer.save_model(training_args.output_dir)
    # logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
    logger.info("Training completed!")
    #############
    # push to hub
    #############
    # if training_args.push_to_hub:
    #     logger.info("Pushing to hub...")
    #     trainer.push_to_hub(**kwargs)



if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
