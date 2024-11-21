import os
import math
from types import MethodType
import torch
from loguru import logger
from transformers import BitsAndBytesConfig
from transformers.utils.versions import require_version
from typing import Tuple
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_kbit_training

try:
    from transformers.integrations import is_deepspeed_zero3_enabled
except ImportError:  # https://github.com/huggingface/transformers/releases/tag/v4.33.1
    from transformers.deepspeed import is_deepspeed_zero3_enabled

from utils.utils import apply_llama_patch, find_all_linear_names, print_trainable_parameters

def setup_model(model_args, script_args, training_args, config_class, model_class):
    """
    Setup and configure the model for training.
    
    Args:
        model_args: Model arguments
        script_args: Script arguments
        training_args: Training arguments
        config_class: Model configuration class
        model_class: Model class
        
    Returns:
        configured model
    """
    if not model_args.model_name_or_path:
        raise ValueError(f"Error, model_name_or_path is None, SFT must be loaded from a pre-trained model")

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    
    # Handle distributed training setup
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size != 1
    if ddp:
        model_args.device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}
        training_args.gradient_accumulation_steps = training_args.gradient_accumulation_steps // world_size or 1
    
    if script_args.qlora and (len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled()):
        logger.warning("FSDP and DeepSpeed ZeRO-3 are both currently incompatible with QLoRA.")

    # Setup model configuration
    config_kwargs = {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }
    config = config_class.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    # Configure RoPE scaling
    if model_args.rope_scaling is not None:
        if hasattr(config, "rope_scaling"):
            if model_args.rope_scaling == "dynamic":
                logger.warning(
                    "Dynamic NTK may not work well with fine-tuning. "
                    "See: https://github.com/huggingface/transformers/pull/24653"
                )
            current_max_length = getattr(config, "max_position_embeddings", None)
            if current_max_length and script_args.model_max_length > current_max_length:
                scaling_factor = float(math.ceil(script_args.model_max_length / current_max_length))
            else:
                logger.warning(f"The model_max_length({script_args.model_max_length}) is smaller than max "
                             f"length({current_max_length}). Consider increase model_max_length.")
                scaling_factor = 1.0
            
            setattr(config, "rope_scaling", {"type": model_args.rope_scaling, "factor": scaling_factor})
            logger.info("Using {} scaling strategy and setting scaling factor to {}".format(
                model_args.rope_scaling, scaling_factor
            ))
        else:
            logger.warning("Current model does not support RoPE scaling.")

    # Configure Flash Attention
    if model_args.flash_attn:
        if is_flash_attn_2_available:
            config_kwargs["use_flash_attention_2"] = True
            logger.info("Using FlashAttention-2 for faster training and inference.")
        else:
            logger.warning("FlashAttention-2 is not installed.")
    elif model_args.shift_attn and getattr(config, "model_type", None) == "llama":
        logger.warning("Using `--flash_attn` for faster training in large context length, enable if your GPU"
                      " is RTX3090, RTX4090, A100 or H100.")

    # Configure shifted sparse attention
    if model_args.shift_attn:
        if getattr(config, "model_type", None) == "llama":
            setattr(config, "group_size_ratio", 0.25)
            apply_llama_patch()
            logger.info("Using shifted sparse attention with group_size_ratio=1/4.")
        else:
            logger.warning("Current model does not support shifted sparse attention.")

    # Configure model quantization
    load_in_4bit = model_args.load_in_4bit
    load_in_8bit = model_args.load_in_8bit
    if load_in_4bit and load_in_8bit:
        raise ValueError("Error, load_in_4bit and load_in_8bit cannot be set at the same time")
    elif load_in_8bit or load_in_4bit:
        logger.info(f"Quantizing model, load_in_4bit: {load_in_4bit}, load_in_8bit: {load_in_8bit}")
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")
        if load_in_8bit:
            config_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
        elif load_in_4bit:
            if script_args.qlora:
                config_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch_dtype,
                )
            else:
                config_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                )

    # Load the model
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        device_map=model_args.device_map,
        **config_kwargs,
    )

    # Fix specific model architectures
    if getattr(config, "model_type", None) == "chatglm" or getattr(config, "model_type", None) == "internlm2":
        setattr(model, "lm_head", model.transformer.output_layer)
        setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])

    # Configure NEFTune if enabled
    if model_args.neft_alpha > 0:
        input_embed = model.get_input_embeddings()
        if isinstance(input_embed, torch.nn.Embedding):
            def noisy_forward(self: torch.nn.Embedding, x: torch.Tensor) -> torch.Tensor:
                embeddings = input_embed.__class__.forward(self, x)
                dims = self.num_embeddings * self.embedding_dim
                mag_norm = model_args.neft_alpha / (dims ** 0.5)
                embeddings += torch.zeros_like(embeddings).uniform_(-mag_norm, mag_norm)
                return embeddings

            input_embed.forward = MethodType(noisy_forward, input_embed)
            logger.info("Using noisy embedding with alpha={:.2f}".format(model_args.neft_alpha))
        else:
            logger.warning("Input embeddings are not normal nn.Embedding, cannot transform into noisy embedding.")

    # Patch Mixtral MOE model if needed
    if getattr(config, "model_type", None) == "mixtral" and is_deepspeed_zero3_enabled():
        require_version("deepspeed>=0.13.0", "To fix: pip install deepspeed>=0.13.0")
        from deepspeed.utils import set_z3_leaf_modules
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

        set_z3_leaf_modules(model, [MixtralSparseMoeBlock])

    return model 

def setup_peft_model(model, script_args, load_in_8bit, load_in_4bit, training_args):
    """
    Setup PEFT (Parameter-Efficient Fine-Tuning) for the model.
    
    Args:
        model: The base model to apply PEFT
        script_args: Script arguments containing PEFT configuration
        load_in_8bit: Whether model is loaded in 8-bit mode
        load_in_4bit: Whether model is loaded in 4-bit mode
        training_args: Training arguments
        
    Returns:
        model with PEFT configuration applied
    """
    logger.info("Fine-tuning method: LoRA(PEFT)")

    # Set fp32 forward hook for lm_head
    output_layer = getattr(model, "lm_head")
    if isinstance(output_layer, torch.nn.Linear) and output_layer.weight.dtype != torch.float32:
        def fp32_forward_post_hook(module: torch.nn.Module, args: Tuple[torch.Tensor], output: torch.Tensor):
            return output.to(torch.float32)
        output_layer.register_forward_hook(fp32_forward_post_hook)

    # Load pre-trained LoRA model if specified
    if script_args.peft_path is not None:
        logger.info(f"Peft from pre-trained model: {script_args.peft_path}")
        model = PeftModel.from_pretrained(model, script_args.peft_path, is_trainable=True)
    else:
        logger.info("Init new peft model")
        if load_in_8bit or load_in_4bit:
            model = prepare_model_for_kbit_training(model, training_args.gradient_checkpointing)

        # Setup target modules for LoRA
        target_modules = script_args.target_modules.split(',') if script_args.target_modules else None
        if target_modules and 'all' in target_modules:
            target_modules = find_all_linear_names(model, int4=load_in_4bit, int8=load_in_8bit)

        # Setup modules to save
        modules_to_save = script_args.modules_to_save
        if modules_to_save is not None:
            modules_to_save = modules_to_save.split(',')

        logger.info(f"Peft target_modules: {target_modules}")
        logger.info(f"Peft lora_rank: {script_args.lora_rank}")

        # Create and apply LoRA config
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=script_args.lora_rank,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            modules_to_save=modules_to_save
        )
        model = get_peft_model(model, peft_config)

    # Convert trainable parameters to float32
    for param in filter(lambda p: p.requires_grad, model.parameters()):
        param.data = param.data.to(torch.float32)

    model.print_trainable_parameters()
    return model

def setup_model_for_training(model_args, script_args, training_args, config_class, model_class):
    """
    Setup and configure the model for training, including PEFT if specified.
    
    Args:
        model_args: Model arguments
        script_args: Script arguments
        training_args: Training arguments
        config_class: Model configuration class
        model_class: Model class
        
    Returns:
        configured model ready for training
    """
    # First setup the base model
    model = setup_model(model_args, script_args, training_args, config_class, model_class)
    
    # Apply PEFT if specified, otherwise use full parameter training
    if script_args.use_peft:
        model = setup_peft_model(
            model=model,
            script_args=script_args,
            load_in_8bit=model_args.load_in_8bit,
            load_in_4bit=model_args.load_in_4bit,
            training_args=training_args
        )
    else:
        logger.info("Fine-tuning method: Full parameters training")
        model = model.float()
        print_trainable_parameters(model)
    
    return model 