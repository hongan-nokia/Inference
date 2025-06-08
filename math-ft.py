import os, sys

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import datasets
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    GenerationConfig,
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    GenerationConfig
)
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model, AdaLoraConfig


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    print(f"trainable model parameters: {trainable_model_params}. All model parameters: {all_model_params} ")
    return trainable_model_params, all_model_params


dataset = datasets.load_dataset('json', data_files="math_10k.json")

model_id = '/data/Models/llama3_8B_Base/'

tokenizer = AutoTokenizer.from_pretrained(model_id, max_new_tokens=1024)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

base_model = LlamaForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=False,
    # device_map='auto',
)

# base_model.gradient_checkpointing_enable()
# model = prepare_model_for_kbit_training(base_model)
model = base_model

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj"],
    task_type="CAUSAL_LM",
)

_, ori_p = print_number_of_trainable_model_parameters(model)
model = get_peft_model(model, peft_config)

peft_p, _ = print_number_of_trainable_model_parameters(model)
print(f"# Trainable Parameter \nBefore: {ori_p} \nAfter: {peft_p} \nPercentage: {round(peft_p / ori_p * 100, 2)}%")

max_length = 1024


def generate_prompt(data_point):
    return f"""
    Below is an instruction that describes a mathematical task. Write a response that appropriately completes the request.  

    ### Instruction:
    {data_point["instruction"]}

    ### Response:
    {data_point["output"]}

    """


def tokenize(tokenizer, prompt, max_length=max_length, add_eos_token=False):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None)
    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < max_length
            and add_eos_token):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    result["labels"] = result["input_ids"].copy()
    return result


def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(tokenizer, full_prompt, add_eos_token=True)
    user_prompt = generate_prompt(data_point)
    tokenized_user_prompt = tokenize(tokenizer, user_prompt)
    user_prompt_len = len(tokenized_user_prompt["input_ids"])
    mask_token = [-100] * user_prompt_len
    tokenized_full_prompt["labels"] = mask_token + tokenized_full_prompt["labels"][user_prompt_len:]
    return tokenized_full_prompt


train_val = dataset["train"].train_test_split(test_size=2000, shuffle=True, seed=42)
train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)

args = TrainingArguments(
    output_dir="lora_math-32",
    num_train_epochs=3,
    gradient_accumulation_steps=1,
    warmup_steps=10,
    fp16=False,
    bf16=False,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    lr_scheduler_type="constant",
    per_device_train_batch_size=5,
    per_device_eval_batch_size=5,
    gradient_checkpointing=False,
    group_by_length=False,
    logging_steps=40,
    logging_dir="./logs",
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=5,
    do_eval=True,
    disable_tqdm=False,
)

if torch.cuda.device_count() > 1:
    model.is_parallelizable = True
    model.model_parallel = True

trainer = Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=args,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
)

model.config.use_cache = False
trainer.train()
model.save_pretrained("lora_math-32/final")
print('done')
