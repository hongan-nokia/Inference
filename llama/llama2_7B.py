# -*- coding: utf-8 -*-
"""
@Time: 1/14/2025 4:30 PM
@Author: Honggang Yuan
@Email: honggang.yuan@nokia-sbell.com
Description: 
"""
import torch
import warnings
import os
import sys
import json
import threading
import readline
import datasets
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from transformers import (
    LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline, AutoTokenizer,
    AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments, GenerationConfig, LlamaModel
)
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model

warnings.filterwarnings("ignore")

base_model_path = '/guozhanqiu/yuan/models/llama2_7B_chat_hf/'

original_model = LlamaForCausalLM.from_pretrained(
    base_model_path,
    device_map='cuda:0',
)
original_model = original_model.eval()
tokenizer = LlamaTokenizer.from_pretrained(base_model_path)

print(original_model)
