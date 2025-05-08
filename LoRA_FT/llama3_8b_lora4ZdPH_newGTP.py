# -*- coding: utf-8 -*-
"""
@Time: 5/5/2025 2:52 PM
@Author: Honggang Yuan
@Email: honggang.yuan@nokia-sbell.com
Description: 
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
torch.cuda.device_count()

from datasets import load_dataset
import torch
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
from datetime import datetime

base_model_id = "/guozhanqiu/LLModels/llama3_8B_Base"
task_id = "task620_ohsumed_medical_subject_headings_answer_generation"
dataset_id = f"/guozhanqiu/Datasets/{task_id}/data"

dataset = load_dataset(dataset_id)
max_length = 1000

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
    text = (
        f"Input: {example['input']} \n\n"
        f"Output: {example['output']}"
    )
    return text


def generate_and_tokenize_prompt(exp_prompt):
    """
    new GTP
    """
    prompt = formatting_func(exp_prompt)

    encodings = tokenizer(
        prompt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encodings['input_ids'].squeeze()
    attention_mask = encodings['attention_mask'].squeeze()

    labels = torch.full_like(input_ids, -100)
    output_text = exp_prompt['output']

    if isinstance(output_text, list):
        combine_strings = lambda lst: "".join(lst)
        output_text2one = combine_strings(output_text)
        output_tokens = tokenizer.encode(output_text2one, add_special_tokens=False)
    elif isinstance(output_text, str):
        output_tokens = tokenizer.encode(output_text, add_special_tokens=False)

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

    for i in range(len(prompt_tokens) - len(output_tokens) + 1):
        if prompt_tokens[i:i + len(output_tokens)] == output_tokens:
            start_idx = i
            end_idx = i + len(output_tokens)
            # 确保索引在 max_length 内
            if end_idx <= max_length:
                labels[start_idx:end_idx] = input_ids[start_idx:end_idx]
            break

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


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
    r=4,
    lora_alpha=16,
    target_modules=zdph_target_modules,
    bias='all',
    lora_dropout=0,
    task_type='CAUSAL_LM'
)

peft_model = get_peft_model(model, lora_config)
print_trainable_parameters(peft_model)

task_name = dataset_id.split('/')[-2].split('_')[0]
project = f'zdph_{task_name}'
base_model_name = 'llama3_8b'
run_name = base_model_name + '-' + project
output_dir = './' + run_name

if torch.cuda.device_count() > 1:
    model.is_parallelizable = True
    model.model_parallel = True

tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=5,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2.5e-5,
        bf16=False,
        optim="paged_adamw_32bit",
        evaluation_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=60,
        save_total_limit=5,
        do_eval=True,
        weight_decay=0.01,
        load_best_model_at_end=True,
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
        # run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d')}",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False

trainer.train()

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ************************************************************************************************************************************************************
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
Instruction:
You are given 2 questions and you need to compare them and understand are they semantically similar or not, by providing explanation and after that label. 0 means dissimilar and 1 means similar.
Question 1: {{question_1}}
Question 2: {{question_2}}
Explanation: {{expandlab}}
"""

from datasets import load_dataset

dataset = load_dataset('borismartirosyan/glue-qqp-sampled-explanation')

EOS_TOKEN = tokenizer.eos_token


def formatting_prompts_func(examples):
    question_1 = examples["question1"]
    question_2 = examples["question2"]
    explanations = examples["explanation"]
    labels = examples["label"]
    texts = []
    for q1, q2, exp, labl in zip(question_1, question_2, explanations, labels):
        # Must add EOS_TOKEN, otherwise your generation will go on forever! BOS token will be added automatically
        text = prompt.replace('{{question_1}}', q1).replace('{{question_2}}', q2).replace("{{expandlab}}", exp + ' label: ' + labl) + EOS_TOKEN
        texts.append(text)
    return {"text": texts, }


dataset = dataset.map(formatting_prompts_func, batched=True)
