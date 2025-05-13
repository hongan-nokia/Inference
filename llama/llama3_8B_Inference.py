# -*- coding: utf-8 -*-
"""
@Time: 8/12/2024 3:57 PM
@Author: Honggang Yuan
@Email: honggang.yuan@nokia-sbell.com
Description: 
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
torch.cuda.device_count()

import torch
from langchain import PromptTemplate, LLMChain
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from transformers import (
    LlamaForCausalLM, AutoModelForCausalLM,
    pipeline,
    AutoTokenizer,
    BertTokenizer, BertModel
)


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


model_dir = "/data/Models/llama3_8B_Base"

tokenizer = AutoTokenizer.from_pretrained(model_dir, truncation=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
# print(tokenizer)

LLM_model = LlamaForCausalLM.from_pretrained(
    model_dir,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map='auto',
)

pipe = pipeline(
    "text-generation",
    model=LLM_model,
    tokenizer=tokenizer,
    max_length=100000,
    temperature=0.05,
    top_p=0.9,
    repetition_penalty=1
)

ans_agent = HuggingFacePipeline(pipeline=pipe)

agent_prompt_template = """
You are an expert scholar in the fields of Computer Science & Artificial Intelligence, proficient in academic writing and rigorous analysis. 
Please respond to the following question in an academic style, grounded in established theoretical frameworks, logical reasoning, and objective evidence: 
{query}. 

Your response must adhere to the following guidelines:

Formal Language: Use precise, objective, and scholarly language, avoiding colloquial expressions or subjective phrasing. 

Rigorous Logic: Ensure the argument is coherent, supported by evidence or logical reasoning, and free from unsubstantiated claims.

Accurate Terminology: Employ field-specific terminology accurately, providing brief explanations of terms where necessary to enhance accessibility.

Comprehensive Content: Address the core aspects of the question, while considering potential counterarguments or alternative perspectives and briefly evaluating their validity.

Word Count: Limit the response to  300-500 words, unless otherwise specified.

If the question involves specific cases, data, or scenarios, provide analysis based on reasonable assumptions, clearly stating any assumptions made. If the question exceeds your knowledge scope, acknowledge the limitation and offer a reasoned response based on available information.

### Response:

"""
prompt_template = PromptTemplate(template=agent_prompt_template, input_variables=["query"])

ScholarAgent = LLMChain(prompt=prompt_template, llm=ans_agent)

q = "What's the differnce between the cache and memory and disk? "
out = ScholarAgent.run(query=q)
print(f"{color.RED}{' Scholar agent answer : ' + out}{color.END}\n")
