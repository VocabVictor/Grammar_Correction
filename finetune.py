# -*- coding: utf-8 -*-
# Copyright 2023 XuMing(xuming624@qq.com) and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, LLaMA, Bloom, ...) on a json file or a dataset.

part of code is modified from https://github.com/shibing624/textgen
"""

import os
from loguru import logger
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed
)
from transformers.trainer_pt_utils import LabelSmoother
from utils.template import get_conv_template
from arguments import DataArguments, ScriptArguments, ModelArguments
from arguments.model import MODEL_CLASSES
from utils.data_utils import setup_tokenizer, load_datasets
from utils.train_utils import (
    prepare_train_dataset, 
    prepare_eval_dataset, 
    setup_trainer,
    train_model,
    evaluate_model
)
from utils.model_utils import setup_model_for_training


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments, ScriptArguments))
    model_args, data_args, training_args, script_args = parser.parse_args_into_dataclasses()

    logger.info(f"Model args: {model_args}")
    logger.info(f"Data args: {data_args}")
    logger.info(f"Training args: {training_args}")
    logger.info(f"Script args: {script_args}")
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_args.model_type]
    # Load tokenizer
    prompt_template = get_conv_template(script_args.template_name)
    tokenizer = setup_tokenizer(model_args, script_args, tokenizer_class, prompt_template)

    IGNORE_INDEX = LabelSmoother.ignore_index if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    # Get datasets
    raw_datasets = load_datasets(data_args, model_args)

    # Prepare training dataset
    train_dataset, max_train_samples = prepare_train_dataset(
        raw_datasets=raw_datasets,
        training_args=training_args,
        data_args=data_args,
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        script_args=script_args,
        IGNORE_INDEX=IGNORE_INDEX
    )

    # Prepare evaluation dataset
    eval_dataset, max_eval_samples = prepare_eval_dataset(
        raw_datasets=raw_datasets,
        training_args=training_args,
        data_args=data_args,
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        script_args=script_args,
        IGNORE_INDEX=IGNORE_INDEX
    )

    # Get world size for distributed training
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size != 1

    # Load and setup model for training
    model = setup_model_for_training(
        model_args=model_args,
        script_args=script_args,
        training_args=training_args,
        config_class=config_class,
        model_class=model_class
    )

    # Setup trainer
    trainer = setup_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        IGNORE_INDEX=IGNORE_INDEX,
        ddp=ddp
    )

    # Execute training
    train_model(
        trainer=trainer,
        training_args=training_args,
        tokenizer=tokenizer,
        model=model,
        max_train_samples=max_train_samples,
        IGNORE_INDEX=IGNORE_INDEX
    )

    # Execute evaluation
    evaluate_model(
        trainer=trainer,
        training_args=training_args,
        max_eval_samples=max_eval_samples
    )

if __name__ == "__main__":
    main()