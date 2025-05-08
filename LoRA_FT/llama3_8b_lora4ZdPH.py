# -*- coding: utf-8 -*-
"""
@Time: 5/5/2025 2:52 PM
@Author: Honggang Yuan
@Email: honggang.yuan@nokia-sbell.com
Description: 
"""
from datasets import load_dataset
import torch
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
from datetime import datetime

base_model_id = "/guozhanqiu/LLModels/llama3_8B_Base"
dataset_id = "/guozhanqiu/Datasets/task280_stereoset_classification_stereotype_type/data"

dataset = load_dataset(dataset_id)
max_length = 40000

# # or load the separate splits if the dataset has train/validation/test splits
train_dataset = load_dataset(dataset_id, split="train")
valid_dataset = load_dataset(dataset_id, split="validation")
test_dataset = load_dataset(dataset_id, split="test")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=False,
    bnb_4bit_user_double_quant=False,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side='left',
    add_eos_token=True,
    ass_bos_token=True
)
tokenizer.pad_token = tokenizer.eos_token


def formatting_func(example):
    text = f"""
    Input: {example['input']}

    Output: {example['output']}
    """
    return text


def generate_and_tokenize_prompt(prompt):
    """
    GTP
    """
    return tokenizer(formatting_func(prompt))


tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = valid_dataset.map(generate_and_tokenize_prompt)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


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


num_layers = model.config.num_hidden_layers  # 总层数
trained_layers = 12
zdph_target_modules = [
    f"model.layers.{i}.self_attn.{proj}"
    for i in range(num_layers - trained_layers, num_layers)
    for proj in ["q_proj", "k_proj", "v_proj"]
]
# vanilla_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=zdph_target_modules,
    bias='all',
    lora_dropout=0,
    task_type='CAUSAL_LM'
)

peft_model = get_peft_model(model, lora_config)
print_trainable_parameters(peft_model)

# ##########################################
# ------------------------------------------
# ##########################################

project = 'zdph_task280'
base_model_name = 'llama3_8b'
run_name = base_model_name + '-' + project
output_dir = './' + run_name

if torch.cuda.device_count() > 1:
    model.is_parallelizable = True
    model.model_parallel = True

tokenizer.pad_token = tokenizer.eos_token
"""
trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=10,
        num_train_epochs=3,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=10,
        gradient_checkpointing=True,
        max_steps=100,
        learning_rate=2.5e-5,
        bf16=False,
        optim="paged_adamw_32bit",
        logging_dir="./logs",
        evaluation_strategy="epoch",
        eval_steps=100,
        save_strategy="best",
        save_steps=100,
        do_eval=True,
        weight_decay=0.01,
        # run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
        # run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d')}",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
"""

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=10,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2.5e-5,
        bf16=False,
        optim="paged_adamw_32bit",
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=5,
        do_eval=True,
        weight_decay=0.01,
        load_best_model_at_end=True,
        # run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
        # run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d')}",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
