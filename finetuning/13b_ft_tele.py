# -*- coding: utf-8 -*-
"""
@Time: 3/5/2025 5:47 PM
@Author: Honggang Yuan
@Email: honggang.yuan@nokia-sbell.com
Description: 
"""
from datasets import load_dataset
import torch
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForSeq2Seq, Qwen2VLForConditionalGeneration
import transformers
from datetime import datetime

max_length = 1024
jsonl_file_path = "./TeleQnA-main/TeleQnA.jsonl"
base_model_id = "/guozhanqiu/yuan/models/llama2_13b_hf/"
project = 'CFT2/'
base_model_name = 'llama2_13b'
run_name = base_model_name + '-' + project
output_dir = './' + run_name
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_user_double_quant=True,
#     bnb_4bit_quant_type='nf4',
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

lora_config = LoraConfig(
    r=32,  # d*r, r*k. 与下游任务相关
    lora_alpha=64,
    target_modules=['q_proj', 'k_proj', 'v_proj'],
    bias='all',
    lora_dropout=0.1,
    task_type='CAUSAL_LM'
)

train_args = transformers.TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    warmup_steps=10,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_checkpointing=True,
    gradient_accumulation_steps=4,
    save_total_limit=5,
    learning_rate=2.5e-5,
    bf16=False,
    optim="paged_adamw_32bit",
    lr_scheduler_type="constant",
    logging_dir="./logs",
    logging_steps=50,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    do_eval=True,
    weight_decay=0.05,
    deepspeed='./ds_config.json',
    run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
)

dataset = load_dataset('json', data_files=jsonl_file_path, split='train')

train_dataset = dataset.shuffle(seed=500).select(range(9500))
eval_dataset = dataset.shuffle(seed=500).select(range(500))


# print(type(train_dataset))

def formatting_func(example):
    text = f"""
    Below is an question about communication, paired with corresponding answer and its further explanation, and the category of answer and explanation.
    ### Question:
    {example["question"]}
    ### Answer:
    {example["answer"]}
    ### Explanation:
    {example["explanation"]}
    ### Category:
    {example["category"]}"""
    return text


# model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map='auto', quantization_config=bnb_config)
# model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map='auto')
model = AutoModelForCausalLM.from_pretrained(base_model_id)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side='left',
    add_eos_token=True,
    ass_bos_token=True
)
tokenizer.pad_token = tokenizer.eos_token


def generate_and_tokenize_prompt(prompt):
    return tokenizer(formatting_func(prompt))


def generate_and_tokenize_prompt2(prompt):
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding='max_length',
    )
    result['labels'] = result['input_ids'].copy()
    return result


tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt2)
tokenized_eval_dataset = eval_dataset.map(generate_and_tokenize_prompt2)

try:
    a = 1
except  Exception  as e:

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


def print_trainable_parameters(model):
    """
    print the number of trainable parameters in the model
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param}"
    )
    print(f"Trainable params percentage: {trainable_params / all_param * 100}%")


model = get_peft_model(model, lora_config)

print_trainable_parameters(model)

if torch.cuda.device_count() > 1:
    model.is_parallelizable = True
    model.model_parallel = True

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    args=train_args,
    data_collator=DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True),
    # data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False

trainer.train()
