from loguru import logger

def preprocess_function(examples, tokenizer, prompt_template, max_length, script_args, IGNORE_INDEX):
    """
    Preprocessing the instruction datasets.
    
    Args:
        examples: Raw examples containing instruction, input and output fields
        tokenizer: Tokenizer for text processing
        prompt_template: Template for prompts
        max_length: Maximum sequence length
        script_args: Script arguments
        IGNORE_INDEX: Index to ignore in loss calculation
        
    Returns:
        Processed dataset with input_ids, attention_mask and labels
    """
    input_ids_list = []
    attention_mask_list = []
    targets_list = []

    for i in range(len(examples['instruction'])):
        instruction = examples['instruction'][i]
        input_text = examples['input'][i]
        output = examples['output'][i]

        # Combine instruction and input text
        if input_text:
            source_text = f"{instruction}\n{input_text}"
        else:
            source_text = instruction

        # Convert to conversation format expected by the template
        messages = [[source_text, output]]
        dialog = prompt_template.get_dialog(messages)

        # Process each turn in the dialog
        input_ids, labels = [], []
        
        # We only have one turn in this case
        source_ids = tokenizer.encode(text=dialog[0], add_special_tokens=True)
        target_ids = tokenizer.encode(text=dialog[1], add_special_tokens=False)

        # Calculate max lengths for source and target
        total_len = len(source_ids) + len(target_ids)
        max_source_len = int(max_length * (len(source_ids) / total_len))
        max_target_len = int(max_length * (len(target_ids) / total_len))

        # Truncate if necessary
        if len(source_ids) > max_source_len:
            source_ids = source_ids[:max_source_len]
        if len(target_ids) > max_target_len - 1:  # leave room for eos token
            target_ids = target_ids[:max_target_len - 1]
            
        # Remove leading/trailing special tokens if needed
        if len(source_ids) > 0 and source_ids[0] == tokenizer.eos_token_id:
            source_ids = source_ids[1:]
        if len(target_ids) > 0 and target_ids[-1] == tokenizer.eos_token_id:
            target_ids = target_ids[:-1]

        # Skip if the combined length is too long
        if len(source_ids) + len(target_ids) + 1 > max_length:
            logger.warning(f"Skipping example {i} as it is too long")
            continue

        # Combine source and target ids
        input_ids = source_ids + target_ids + [tokenizer.eos_token_id]
        
        # Create labels based on training configuration
        if script_args.train_on_inputs:
            labels = input_ids.copy()
        else:
            labels = [IGNORE_INDEX] * len(source_ids) + target_ids + [tokenizer.eos_token_id]

        input_ids_list.append(input_ids)
        attention_mask_list.append([1] * len(input_ids))
        targets_list.append(labels)

    return dict(
        input_ids=input_ids_list,
        attention_mask=attention_mask_list,
        labels=targets_list,
    )

def filter_empty_labels(example, IGNORE_INDEX):
    """Remove empty labels dataset."""
    return not all(label == IGNORE_INDEX for label in example["labels"]) 