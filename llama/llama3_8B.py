# -*- coding: utf-8 -*-
"""
@Time: 1/13/2025 8:25 PM
@Author: Honggang Yuan
@Email: honggang.yuan@nokia-sbell.com
Description: 
"""
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
    LlamaForCausalLM,
    pipeline,
    AutoTokenizer,
    LlamaModel,
    LlamaConfig
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
base_model = LLM_model
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
print(base_model)
pipe = pipeline(
    "text-generation",
    model=base_model,
    tokenizer=tokenizer,
    max_length=256,
    temperature=0.05,
    top_p=0.9,
    repetition_penalty=1
)

task_agent = HuggingFacePipeline(pipeline=pipe)

analyzer_agent_prompt_template = """
You are an expert of wireless communication system. Your task is to judge whether the current system's performance can meet the user's transmission request.

### System performance: {system_performance}
### Transmission request: {request}

Your response should be in JSON format. The key 'need_to_improve_performance' is a bool. It indicates whether the system need to improve the performance. The key 'reason' explains the reason.

### Response:

"""
analyzer_prompt_template = PromptTemplate(template=analyzer_agent_prompt_template, input_variables=["system_performance", "request"])
AnalyzerAgent = LLMChain(prompt=analyzer_prompt_template, llm=task_agent)

monitor_agent_prompt_template = """
You are an expert of wireless communication system. Your task is to compare the old system state and the current system state.

# Old system state: {old_system_state}
# Current system state: {new_system_state}

Your response should be in JSON format. The key 'need_action' is a bool. It indicates whether the system need to take action due to the changing of system state. The key 'reason' explains the reason. The key 'message' indicates the message conclude the changing part.

# Response:

"""

monitor_prompt_template = PromptTemplate(template=monitor_agent_prompt_template, input_variables=["old_system_state", "new_system_state"])
MonitorAgent = LLMChain(prompt=monitor_prompt_template, llm=task_agent)

orchestrator_agent_prompt_template = """
You are an expert in wireless communication developed by Nokia Bell Labs China. Below is a query that describes a task of communication. Please give your response.

### Query：
{query}

### Response:

"""
orchestrator_prompt_template = PromptTemplate(template=orchestrator_agent_prompt_template, input_variables=["query"])
OrchestratorAgent = LLMChain(prompt=orchestrator_prompt_template, llm=task_agent)

emb_func = SentenceTransformerEmbeddings(model_name=emb_model_dir)
RAGDB = Chroma(
    persist_directory=rag_db,
    embedding_function=emb_func,
)

if __name__ == '__main__':
    LOCAL_HOST = '192.168.1.2'  # 本地地址
    LOCAL_PORT = 12307  # 端口号（自定义）
    REMOTE_HOST = '192.168.1.2'  # 远程地址
    REMOTE_PORT = 12300  # 端口号（自定义）
    udp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_server_socket.bind((LOCAL_HOST, LOCAL_PORT))
    udp_server_socket.setblocking(False)

    while True:
        data = ""
        try:
            recv_data, addr = udp_server_socket.recvfrom(4096)
        except Exception as e:
            continue
        if recv_data:
            recv_data = recv_data.decode("utf-8")
            recv_json = json.loads(recv_data)
            print(f">>>>>>>>>>>>>>>>>... YUNBO_SERVER ... recv_json: {recv_json}")
            if recv_json['agent'] == 'rag':
                rag_ans = ""
                user_query = recv_json['query']
                rag_docs = RAGDB.similarity_search_with_score(user_query, k=1)
                score = rag_docs[0][1]
                if score <= 1.4:
                    rag_ans = rag_docs[0][0].page_content
                print(f"{' RAG result : ' + rag_ans}")
                rag_dict = {"text": rag_ans}
                rag_response = json.dumps(rag_dict)
                udp_server_socket.sendto(rag_response.encode('utf-8'), (REMOTE_HOST, REMOTE_PORT))

            elif recv_json['agent'] == 'monitor_agent':
                monitor_out = MonitorAgent.run(old_system_state=recv_json['old_system_state'], new_system_state=recv_json['new_system_state'])
                print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                print(f"{color.BLUE}{' MonitorAgent answer : ' + monitor_out}{color.END}\n")
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                monitor_dict = {"text": monitor_out, "agent": "monitor_agent"}
                monitor_response = json.dumps(monitor_dict)
                udp_server_socket.sendto(monitor_response.encode('utf-8'), (REMOTE_HOST, REMOTE_PORT))

            elif recv_json['agent'] == 'analyzer_agent':
                analyzer_out = AnalyzerAgent.run(request=recv_json['request'], system_performance=recv_json['sys_performance'])
                print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                print(f"{color.GREEN}{' AnalyzerAgent answer : ' + analyzer_out}{color.END}\n")
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                analyzer_dict = {"text": analyzer_out, "agent": "analyzer_agent"}  # matlab show, task_out is a dict obj
                analyzer_response = json.dumps(analyzer_dict)
                udp_server_socket.sendto(analyzer_response.encode('utf-8'), (REMOTE_HOST, REMOTE_PORT))

            elif recv_json['agent'] == 'orchestrator_agent':
                orchestrator_out = OrchestratorAgent.run(query=recv_json['query'])
                print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                print(f"{color.RED}{' Orchestrator agent answer : ' + orchestrator_out}{color.END}\n")
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                orchestrator_dict = {"text": orchestrator_out, "agent": "orchestrator_agent"}  # matlab show, task_out is a dict obj
                orchestrator_response = json.dumps(orchestrator_dict)
                udp_server_socket.sendto(orchestrator_response.encode('utf-8'), (REMOTE_HOST, REMOTE_PORT))
