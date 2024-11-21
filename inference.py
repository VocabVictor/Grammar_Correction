# 导入所需的库
import os
from typing import List
import sys
import fire
import torch
import transformers
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
from torch.amp import autocast

from peft.utils import prepare_model_for_kbit_training
from peft import (
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM,AutoTokenizer

from utils.prompter import Prompter

from rich.console import Console
from rich.traceback import install

# 初始化控制台和异常处理
console = Console()
install(console=console, show_locals=True, width=120, word_wrap=True)

# 在文件开头添加以下环境变量设置
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 解决tokenizer并行化警告

def train(
    use_lora: bool = True,
    # 模型和数据参数
    base_model: str = "../models/Llama-2-7b-chat-hf",  # 必需的基础模型路径
    data_path: str = "pseudo_data/instruction.json",  # 训练数据路径
    dev_data_path: str = "pseudo_data/nacgec_dev_instruct_zh.json",  # 验证数据路径
    output_dir: str = "./saved_model",  # 模型保存路径
    val_set_size: int =500,  # 验证集大小
    # 训练超参数
    batch_size: int = 4,  # 总批次大小
    micro_batch_size: int = 1,  # 单个批次大小
    num_epochs: int = 5,  # 训练轮数增加到5轮
    learning_rate: float = 2e-5,  # 学习率
    cutoff_len: int = 256,  # 序列最大长度
    # LoRA超参数
    lora_r: int = 8,  # LoRA秩
    lora_alpha: int = 16,  # LoRA缩放因子
    lora_dropout: float = 0.05,  # LoRA dropout率
    lora_target_modules: List[str] = [
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "down_proj", 
        "up_proj"
    ],  # 需要应用LoRA的模块
    # LLM超参数
    train_on_inputs: bool = False,  # 是否在输入上训练
    add_eos_token: bool = False,  # 是否添加结束符
    group_by_length: bool = False,  # 是否按长度分组
    # wandb参数
    wandb_project: str = "glm-4-finetune",  # wandb项目名
    wandb_run_name: str = "",  # wandb运行名
    wandb_watch: str = "gradients",  # wandb监控类型
    wandb_log_model: str = "false",  # 是否记录模型
    resume_from_checkpoint: str = '',  # 恢复训练的检查点路径
    prompt_template_name: str = "phoenix",  # 提示模板名称
):
    try:
        # 计算梯度累积步数
        gradient_accumulation_steps = batch_size // micro_batch_size

        # 初始化提示模板
        prompter = Prompter(prompt_template_name)

        # 设置设备映射
        device_map = "auto"
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        ddp = world_size != 1
        if ddp:
            device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
            gradient_accumulation_steps = gradient_accumulation_steps // world_size

        # 直接加载模型，不使用量化
        if use_lora:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                trust_remote_code=True
            )

        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

        tokenizer.pad_token = tokenizer.eos_token  # 使用eos_token作为pad_token
        tokenizer.padding_side = "left"  # 允许批处理推理

        def tokenize(prompt, add_eos_token=True):
            # there's probably a way to do this with the tokenizer settings
            # but again, gotta move fast
            result = tokenizer(
                prompt,
                truncation=True,
                max_length=cutoff_len,
                padding=False,
                return_tensors=None,
            )
            if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
            ):
                result["input_ids"].append(tokenizer.eos_token_id)
                result["attention_mask"].append(1)

            result["labels"] = result["input_ids"].copy()

            return result

        def generate_and_tokenize_prompt(data_point):
            """
            生成提示并进行分词
            """
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                data_point["input"],
                data_point["output"],
            )
            tokenized_full_prompt = tokenize(full_prompt)
            if not train_on_inputs:
                user_prompt = prompter.generate_prompt(
                    data_point["instruction"], data_point["input"]
                )
                tokenized_user_prompt = tokenize(
                    user_prompt, add_eos_token=add_eos_token
                )
                user_prompt_len = len(tokenized_user_prompt["input_ids"])

                if add_eos_token:
                    user_prompt_len -= 1

                tokenized_full_prompt["labels"] = [
                    -100
                ] * user_prompt_len + tokenized_full_prompt["labels"][
                    user_prompt_len:
                ]  # could be sped up, probably
            return tokenized_full_prompt

        # 配置LoRA
        if use_lora:
            model = prepare_model_for_kbit_training(model)

            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                inference_mode=False,  # 确保训练模式
            )
            model = get_peft_model(model, config)
            
            # 打印可训练参数
            model.print_trainable_parameters()

        # 加载数据集
        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            data = load_dataset("json", data_files=data_path)
        else:
            data = load_dataset(data_path)
        
        if dev_data_path.endswith(".json") or dev_data_path.endswith(".jsonl"):
            dev_data = load_dataset("json", data_files=dev_data_path)
        else:
            dev_data = load_dataset(dev_data_path)

        if val_set_size > 0:
            train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
            # val_data = None
            val_data = dev_data['train'].shuffle().map(generate_and_tokenize_prompt)
        else:
            train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
            val_data = None

        # 从检查点恢复训练
        if resume_from_checkpoint:
            # 检查可用的权重并加载它们
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "pytorch_model.bin"
            )  # 完整检查点
            if not os.path.exists(checkpoint_name):
                checkpoint_name = os.path.join(
                    resume_from_checkpoint, "adapter_model.bin"
                )  # 仅LoRA模型
                resume_from_checkpoint = False  # 防止训练器尝试加载其状态
                
            if use_lora:
                if os.path.exists(checkpoint_name):
                    print(f"Restarting from {checkpoint_name}")
                    adapters_weights = torch.load(checkpoint_name)
                    set_peft_model_state_dict(model, adapters_weights)
                else:
                    print(f"Checkpoint {checkpoint_name} not found")

        # 多GPU设置
        if not ddp and torch.cuda.device_count() > 1:
            # 防止Trainer有多个GPU时尝试自己的DataParallelism
            model.is_parallelizable = True
            model.model_parallel = True

        # 模型配置
        model.config.use_cache = False
        model.config.pretraining_tp = 1

        # wandb配置
        use_wandb = len(wandb_project) > 0
        if use_wandb:
            import wandb
            
            # 设置环境变量
            if len(wandb_project) > 0:
                os.environ["WANDB_PROJECT"] = wandb_project
            if len(wandb_watch) > 0:
                os.environ["WANDB_WATCH"] = wandb_watch

            # 初始化wandb，增加timeout设置
            wandb.init(
                settings=wandb.Settings(init_timeout=300),  # 增加timeout到300秒
                project=wandb_project,
                name=wandb_run_name if wandb_run_name else None,
                config={
                    "learning_rate": learning_rate,
                    "epochs": num_epochs,
                    "batch_size": batch_size,
                    "micro_batch_size": micro_batch_size,
                    "model": base_model,
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                    "lora_dropout": lora_dropout,
                    "lora_target_modules": lora_target_modules,
                    "train_on_inputs": train_on_inputs,
                    "add_eos_token": add_eos_token,
                    "group_by_length": group_by_length,
                }
            )

        # 在 trainer 初始化之前添加
        model.gradient_checkpointing_enable({"use_reentrant": False})

        trainer = Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=4,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                logging_steps=5,
                optim="adamw_torch",
                evaluation_strategy="steps",
                eval_steps=100,
                save_strategy="steps",
                save_steps=100,
                output_dir=output_dir,
                save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=group_by_length,
                report_to="wandb" if use_wandb else None,
                run_name=wandb_run_name if use_wandb else None,
                torch_compile=False,
                bf16=True,
                fp16=False,
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=30)]
        )
        
        model.config.use_cache = False

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
        
        trainer.train()

        # 保存模型为.bin文件
        model.save_pretrained(output_dir, file_name="adapter_model.bin")

        # 训练结束后关闭wandb
        if use_wandb:
            wandb.finish()

    except Exception as e:
        console.print_exception()
        raise e


if __name__ == "__main__":
    fire.Fire(train)
