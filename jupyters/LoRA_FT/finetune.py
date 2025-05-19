# -*- coding: utf-8 -*-
"""
@Time: 5/13/2025 10:25 AM
@Author: Honggang Yuan
@Email: honggang.yuan@nokia-sbell.com
Description:

    Refer to the code of DoRA·······

"""
import argparse
import os
import sys
from typing import List, Optional, Union

# import fire
import torch
import transformers
from datasets import load_dataset
import yaml

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, AutoModel


def load_config(config_path: str):
    with open(config_path, "r") as f:
        y_config = yaml.safe_load(f)
    return y_config


def flatten_config(nest_config):
    """Flatten nested config dictionary."""
    f_config = {}
    for section in nest_config:
        for key, value in nest_config[section].items():
            f_config[key] = value
    return f_config


def validate_config(val_config: dict):
    """Validate that all required parameters are present in the config."""
    required_fields = [
        'base_model',
        'data_path',
        'output_dir',
        # Add other required fields if they must not be empty or None
    ]
    for field in required_fields:
        if field not in val_config or val_config[field] is None or val_config[field] == '':
            raise ValueError(f"Missing or invalid required config field: {field}")


def print_trainable_parameters(trained_model):
    """
    print the number of trainable parameters in the model
    """
    trainable_params = 0
    all_param = 0
    for _, param in trained_model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params:e} || all params: {all_param:e} || trainable params percent: {trainable_params / all_param} %"
    )


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}

                ### Input:
                {data_point["input"]}

                ### Response:
                {data_point["output"]}"""  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}

                ### Response:
                {data_point["output"]}"""  # noqa: E501


def train(
        # model/data params
        base_model: str = "",
        data_path: str = "",
        output_dir: str = "",
        adapter_name: str = "lora",
        load_8bit: bool = False,
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.0,
        cutoff_len: int = 256,
        val_set_size: int = 2000,
        use_gradient_checkpointing: bool = False,
        eval_step: int = 200,
        save_step: int = 200,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_layers: int = 0,
        lora_dropout: float = 0.05,
        # bottleneck adapter hyperparams
        bottleneck_size: int = 256,
        non_linearity: str = "tanh",
        adapter_dropout: float = 0.0,
        use_parallel_adapter: bool = False,
        use_adapterp: bool = False,
        target_modules: List[str] = None,
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    print(
        f"Finetuning model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"adapter_name: {adapter_name}\n"
        f"load_8bit: {load_8bit}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"weight_decay: {weight_decay}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"--------------------------------------------\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_layers: {lora_layers}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"--------------------------------------------\n"
        f"bottleneck_size: {bottleneck_size}\n"
        f"non_linearity: {non_linearity}\n"
        f"adapter_dropout: {adapter_dropout}\n"
        f"use_parallel_adapter: {use_parallel_adapter}\n"
        f"use_adapterp: {use_adapterp}\n"
        f"target_modules: {target_modules}\n"
        f"--------------------------------------------\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"--------------------------------------------\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    device_map = 'auto'

    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if load_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )

    if model.config.model_type == 'llama':
        if 'llama3' in base_model or 'llama-3' in base_model:
            print("load llama-3 tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(base_model)
        else:
            tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allowing batched inference

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) > cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)

            if "chatglm" not in base_model:
                result["attention_mask"].append(1)

        result['labels'] = result['input_ids'].copy()

        if "chatglm" in base_model:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:  # default True
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt['input_ids'])
            tokenized_full_prompt["labels"] = ([-100] * user_prompt_len +
                                               tokenized_full_prompt["labels"][user_prompt_len:])
        return tokenized_full_prompt

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    print(model)

    print("LoRA init...")
    if lora_layers > 0:
        lora_modules = [
            f"model.layers.{i}.self_attn.{proj}"
            for i in range(lora_layers)
            for proj in target_modules
        ]
    elif lora_layers < 0:
        num_layers = model.config.num_hidden_layers
        lora_modules = [
            f"model.layers.{i}.self_attn.{proj}"
            for i in range(num_layers + lora_layers, num_layers)
            for proj in target_modules
        ]
    else:
        lora_modules = target_modules
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    if data_path.endswith(".json"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    print_trainable_parameters(model)

    if val_set_size > 0:
        train_val = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            fp16=True,
            logging_steps=10,
            optim='adamw_torch',
            eval_strategy='steps' if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_step if val_set_size > 0 else None,
            save_steps=save_step,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict

    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.train()

    model.save_pretrained(output_dir)
    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finetune script with YAML config")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file')
    parser.add_argument('--base_model', type=str, default=None, help='Override base_model in YAML config')
    parser.add_argument('--lora_layers', type=int, default=32, help='Override lora_layers in YAML config')
    parser.add_argument('--output_dir', type=str, default=None, help='Override output_dir in YAML config')
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        flat_config = flatten_config(config)
        validate_config(flat_config)

        # Override YAML config with command-line arguments if provided
        if args.base_model is not None:
            flat_config['base_model'] = args.base_model
        if args.lora_layers is not None:
            flat_config['lora_layers'] = args.lora_layers
        if args.output_dir is not None:
            flat_config['output_dir'] = args.output_dir

        # Re-validate config after overrides
        validate_config(flat_config)
        train(**flat_config)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
