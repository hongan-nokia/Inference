# -*- coding: utf-8 -*-
"""
@Time: 1/13/2025 8:25 PM
@Author: Honggang Yuan
@Email: honggang.yuan@nokia-sbell.com
Description: 
"""
import numpy as np
import transformers

# -*- coding: utf-8 -*-
"""
@Time: 8/12/2024 3:57 PM
@Author: Honggang Yuan
@Email: honggang.yuan@nokia-sbell.com
Description: 
"""
import json
import socket

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


rag_db = "/home/fa/6G_Test_LLM_0819/Database/chroma/task_requirement_db"
emb_model_dir = "/home/fa/6G_Test_LLM_0819/all-MiniLM-L6-v2-main"
task_model_dir = "/home/fa/6G_Test_LLM_0819/6G_test_agent"

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=task_model_dir,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="auto",
#     max_length=512,
# )
#
# pipeline("Hey how are you doing today?")

tokenizer = AutoTokenizer.from_pretrained(task_model_dir, truncation=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
print(tokenizer)

LLM_model = LlamaForCausalLM.from_pretrained(
    task_model_dir,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map='auto',
)

pipe = pipeline(
    "text-generation",
    model=LLM_model,
    tokenizer=tokenizer,
    max_length=256,
    temperature=0.05,
    top_p=0.9,
    repetition_penalty=1
)

task_agent = HuggingFacePipeline(pipeline=pipe)

orchestrator_agent_prompt_template = """
You are an expert in wireless communication developed by Nokia Bell Labs China. Below is a query that describes a task of communication. Please give your response.

### Query：
{query}

### Response:

"""
orchestrator_prompt_template = PromptTemplate(template=orchestrator_agent_prompt_template, input_variables=["query"])
OrchestratorAgent = LLMChain(prompt=orchestrator_prompt_template, llm=task_agent)

if __name__ == '__main__':
    q = ""
    orchestrator_out = OrchestratorAgent.run(query=q)
    print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(f"{color.RED}{' Orchestrator agent answer : ' + orchestrator_out}{color.END}\n")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    orchestrator_dict = {"text": orchestrator_out, "agent": "orchestrator_agent"}  # matlab show, task_out is a dict obj
