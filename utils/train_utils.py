from loguru import logger
import torch
from transformers import DataCollatorForSeq2Seq
from utils.preprocessing import preprocess_function, filter_empty_labels
from utils.trainer import SavePeftModelTrainer
from utils.utils import save_model, save_model_zero3
try:
    from transformers.integrations import is_deepspeed_zero3_enabled
except ImportError:
    from transformers.deepspeed import is_deepspeed_zero3_enabled
import math

def prepare_train_dataset(raw_datasets, training_args, data_args, tokenizer, prompt_template, script_args, IGNORE_INDEX):
    """
    Prepare training dataset with preprocessing and filtering.
    
    Args:
        raw_datasets: The raw datasets loaded from source
        training_args: Training arguments
        data_args: Data arguments
        tokenizer: Tokenizer for text processing
        prompt_template: Template for prompts
        script_args: Script arguments
        IGNORE_INDEX: Index to ignore in loss calculation
        
    Returns:
        tuple: (train_dataset, max_train_samples)
    """
    train_dataset = None
    max_train_samples = 0
    
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
            
        train_dataset = raw_datasets['train'].shuffle(seed=42)
        max_train_samples = len(train_dataset)
        
        if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
            
        logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")
        
        with training_args.main_process_first(desc="Train dataset tokenization"):
            train_dataset = train_dataset.shuffle().map(
                lambda x: preprocess_function(x, tokenizer, prompt_template, script_args.model_max_length, 
                                           script_args, IGNORE_INDEX),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
            
            train_dataset = train_dataset.filter(
                lambda x: filter_empty_labels(x, IGNORE_INDEX),
                num_proc=data_args.preprocessing_num_workers
            )
            
            logger.debug(f"Num train_samples: {len(train_dataset)}")
            logger.debug("Tokenized training example:")
            logger.debug(f"Decode input_ids[0]:\n{tokenizer.decode(train_dataset[0]['input_ids'])}")
            replaced_labels = [label if label != IGNORE_INDEX else tokenizer.pad_token_id
                             for label in list(train_dataset[0]['labels'])]
            logger.debug(f"Decode labels[0]:\n{tokenizer.decode(replaced_labels)}")
            
    return train_dataset, max_train_samples

def prepare_eval_dataset(raw_datasets, training_args, data_args, tokenizer, prompt_template, script_args, IGNORE_INDEX):
    """
    Prepare evaluation dataset with preprocessing and filtering.
    
    Args:
        raw_datasets: The raw datasets loaded from source
        training_args: Training arguments
        data_args: Data arguments
        tokenizer: Tokenizer for text processing
        prompt_template: Template for prompts
        script_args: Script arguments
        IGNORE_INDEX: Index to ignore in loss calculation
        
    Returns:
        tuple: (eval_dataset, max_eval_samples)
    """
    eval_dataset = None
    max_eval_samples = 0
    
    if training_args.do_eval:
        with training_args.main_process_first(desc="Eval dataset tokenization"):
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
                
            eval_dataset = raw_datasets["validation"]
            max_eval_samples = len(eval_dataset)
            
            if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
                
            eval_size = len(eval_dataset)
            logger.debug(f"Num eval_samples: {eval_size}")
            
            if eval_size > 500:
                logger.warning(f"Num eval_samples is large: {eval_size}, "
                             f"training slow, consider reduce it by `--max_eval_samples=50`")
                             
            logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")
            
            eval_dataset = eval_dataset.map(
                lambda x: preprocess_function(x, tokenizer, prompt_template, script_args.model_max_length,
                                           script_args, IGNORE_INDEX),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=eval_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
            
            eval_dataset = eval_dataset.filter(
                lambda x: filter_empty_labels(x, IGNORE_INDEX),
                num_proc=data_args.preprocessing_num_workers
            )
            
            logger.debug(f"Num eval_samples: {len(eval_dataset)}")
            logger.debug("Tokenized eval example:")
            logger.debug(tokenizer.decode(eval_dataset[0]['input_ids']))
            
    return eval_dataset, max_eval_samples

def setup_trainer(model, training_args, train_dataset, eval_dataset, tokenizer, IGNORE_INDEX, ddp=False):
    """
    Setup and configure the trainer.
    
    Args:
        model: The model to train
        training_args: Training arguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        tokenizer: Tokenizer for text processing
        IGNORE_INDEX: Index to ignore in loss calculation
        ddp: Whether using distributed data parallel
        
    Returns:
        configured trainer
    """
    # Setup gradient checkpointing
    if training_args.gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):
        def _enable_gradient_checkpointing(module):
            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable()
                module.config.use_cache = False
        
        model.apply(_enable_gradient_checkpointing)
        logger.info("Gradient checkpointing enabled.")
    else:
        model.config.use_cache = True
        logger.info("Gradient checkpointing disabled.")
    
    # Enable input gradients
    model.enable_input_require_grads()
    
    # Configure model parallelism
    if not ddp and torch.cuda.device_count() > 1:
        # Keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    # Setup data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=IGNORE_INDEX,
        pad_to_multiple_of=4 if tokenizer.padding_side == "right" else None,  # for shifted sparse attention
    )

    # Initialize trainer
    trainer = SavePeftModelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    return trainer 

def train_model(trainer, training_args, tokenizer, model, max_train_samples, IGNORE_INDEX):
    """
    Execute model training process.
    
    Args:
        trainer: The configured trainer
        training_args: Training arguments
        tokenizer: Tokenizer for text processing
        model: The model to train
        max_train_samples: Maximum number of training samples
        IGNORE_INDEX: Index to ignore in loss calculation
    """
    if training_args.do_train:
        logger.info("*** Train ***")
        
        # Log training dataloader example if main process
        if trainer.is_world_process_zero():
            sample = next(iter(trainer.get_train_dataloader()))
            logger.debug(f"Train dataloader example: {sample}")
            logger.debug(f"input_ids:\n{list(sample['input_ids'])[:3]}, \nlabels:\n{list(sample['labels'])[:3]}")
            logger.debug(f"Decode input_ids[0]:\n{tokenizer.decode(sample['input_ids'][0])}")
            replaced_labels = [label if label != IGNORE_INDEX else tokenizer.pad_token_id 
                             for label in sample['labels'][0]]
            logger.debug(f"Decode labels[0]:\n{tokenizer.decode(replaced_labels)}")
        
        # Resume from checkpoint if specified
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
            
        # Start training
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        # Log and save metrics
        metrics = train_result.metrics
        metrics["train_samples"] = max_train_samples
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # Post-training configuration
        model.config.use_cache = True  # enable cache after training
        tokenizer.padding_side = "left"  # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"

        # Save model if main process
        if trainer.is_world_process_zero():
            logger.debug(f"Training metrics: {metrics}")
            logger.info(f"Saving model checkpoint to {training_args.output_dir}")
            if is_deepspeed_zero3_enabled():
                save_model_zero3(model, tokenizer, training_args, trainer)
            else:
                save_model(model, tokenizer, training_args) 

def evaluate_model(trainer, training_args, max_eval_samples):
    """
    Execute model evaluation process.
    
    Args:
        trainer: The configured trainer
        training_args: Training arguments
        max_eval_samples: Maximum number of evaluation samples
        
    Returns:
        evaluation metrics
    """
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")

        metrics["eval_samples"] = max_eval_samples
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        if trainer.is_world_process_zero():
            logger.debug(f"Eval metrics: {metrics}")
            
        return metrics