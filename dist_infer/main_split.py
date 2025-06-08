# --------------------------------------------------------
#  main_split_fixed_v5.py
#
#  说明：
#   1. 假设你拆分成：前 12 层 → stage1；后 20 层 → stage2，共 32 层 Llama 模型。
#   2. 先加载完整的 HF LlamaForCausalLM，然后从中“剥离”子模块，
#      避免直接构造 LlamaDecoderLayer 导致 rotary‐emb 返回 None 的问题。
#   3. 在调用每个 LlamaDecoderLayer 时不传 position_ids，依赖它内部“根据 hidden_states 生成 1D position_ids”的默认逻辑。
#   4. 手动构造 fcauseal_mask（bool 下三角 + padding），传给 attention_mask 参数即可。
#
#  你只需修改下面两行路径，就能直接运行：
#    base_model         : 指向 llama3_8B_Base 目录（包含 config.json/pytorch_model.bin/tokenizer 等）
#    split_weights_dir  : 指向存放 model_stage1.safetensors、model_stage2.safetensors 的文件夹
# --------------------------------------------------------

import os
import torch
from torch import nn
from safetensors.torch import load_file
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

# ——————————————————————————————————————————————————————————————————————————————————
# 【请在此处修改为你本地实际的路径】
base_model         = "/guozhanqiu/LLModels/llama3_8B_Base"
split_weights_dir  = "/guozhanqiu/model_layers"
stage1_path        = os.path.join(split_weights_dir, "model_stage1.safetensors")
stage2_path        = os.path.join(split_weights_dir, "model_stage2.safetensors")
# ——————————————————————————————————————————————————————————————————————————————————

# ——————————————————————————————————————————————————————————————————————————————————
# 1. 加载完整的 LlamaForCausalLM（仅用于“剥离”子模块结构）
#    用 float16 且先放到 CPU，以节省显存
# ——————————————————————————————————————————————————————————————————————————————————
print("1️⃣ 载入完整的 LlamaForCausalLM（仅用于获取子模块结构）……")
full_model = LlamaForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="cpu",
    low_cpu_mem_usage=True
)
config = full_model.config
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
print(f"   → 模型共有 {config.num_hidden_layers} 层 Transformer（应为 32）。")

# ——————————————————————————————————————————————————————————————————————————————————
# 2. 明确拆分层数：前 12 层属于 Stage1，后（config.num_hidden_layers − 12）层属于 Stage2
# ——————————————————————————————————————————————————————————————————————————————————
client_layers = 12
assert config.num_hidden_layers == 32, "Expect num_hidden_layers=32"
server_layers = config.num_hidden_layers - client_layers  # 32 - 12 = 20
print(f"2️⃣ 拆分：前 {client_layers} 层 → Stage1；后 {server_layers} 层 → Stage2。")

# 2.1 从 full_model 拆出 embed_tokens
embed_tokens = full_model.model.embed_tokens

# 2.2 从 full_model 拆出前 client_layers 个 decoder 层（layers[0..11]）
stage1_layers = [full_model.model.layers[i] for i in range(client_layers)]

# 2.3 从 full_model 拆出后 server_layers 个 decoder 层（layers[12..31]）
stage2_layers = [full_model.model.layers[i] for i in range(client_layers, config.num_hidden_layers)]

# 2.4 从 full_model 拆出最后的 RMSNorm
rms_norm = full_model.model.norm

# 2.5 从 full_model 拆出 lm_head
lm_head = full_model.lm_head

# 释放 full_model 以节省内存，保留子模块引用
del full_model

# ——————————————————————————————————————————————————————————————————————————————————
# 3. 定义 Stage1/Stage2 子模块 Wrapper 类
#    - 在 forward 时**不传 position_ids**，让 HF 官方内部自动处理 rotary embedding 的 position_ids
#    - 手动构造 bool 型的 causal_mask（下三角 + padding），并传给 attention_mask
# ——————————————————————————————————————————————————————————————————————————————————
class LlamaStage1(nn.Module):
    """
    Stage1：embed_tokens + 前 client_layers 层 LlamaDecoderLayer
    """
    def __init__(self, embed_tokens: nn.Embedding, layers: nn.ModuleList):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.layers = nn.ModuleList(layers)

    def forward(self,
                input_ids: torch.LongTensor,      # [B, L]
                attention_mask: torch.Tensor,     # [B, L]
                past_key_values=None,
                return_dict: bool = False,
                output_attentions: bool = False,
                use_cache: bool = False
               ):
        """
        复制 Hugging Face LlamaModel.forward 的流程，只跑到第 client_layers 层结束。
        主动**不传 position_ids**，让 LlamaDecoderLayer 内部自动构建 1D position_ids。
        返回：last_hidden_state, past_key_values (若 use_cache), hidden_states/attns (若输出它们),
              causal_mask, seq_len (为下游 Stage2 用)
        """
        # 1) embed_tokens
        hidden_states = self.embed_tokens(input_ids)  # [B, L, H]

        # 2) 准备 bool 型的 4D 下三角因果掩码
        B, L = input_ids.shape
        #    下三角 [L, L]，dtype=torch.bool
        tril = torch.tril(torch.ones((L, L), dtype=torch.bool, device=input_ids.device))  # [L, L]
        #    扩展到 [B, 1, L, L]
        causal_mask = tril.unsqueeze(0).unsqueeze(1).expand(B, 1, L, L)                  # [B, 1, L, L]
        #    padding_mask：attention_mask.bool() → [B, L] → reshape 成 [B, 1, 1, L]
        padding_mask = attention_mask.bool().view(B, 1, 1, L)                            # [B, 1, 1, L]
        #    用“逻辑与”屏蔽 padding
        causal_mask = causal_mask & padding_mask                                         # [B, 1, L, L]

        all_hidden_states = () if output_attentions or return_dict else None
        all_self_attns   = () if output_attentions else None
        next_cache       = None

        # 3) 逐层过前 client_layers 层；**不传 position_ids, cache_position**
        for layer in self.layers:
            if output_attentions:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,   # 传入 bool 型的 causal_mask
                output_attentions=output_attentions,
                use_cache=use_cache
            )
            # layer_outputs[0] 始终是更新后的 hidden_states
            hidden_states = layer_outputs[0]  # [B, L, H]
            if use_cache:
                # 若不输出 attn，那 layer_outputs[1] 就是 present_key_value
                next_cache = layer_outputs[1] if not output_attentions else layer_outputs[2]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # 返回结果：我们额外需要把 causal_mask 和当前序列长度 L 传下去
        if return_dict:
            return {
                "last_hidden_state": hidden_states,
                "past_key_values": next_cache,
                "hidden_states": all_hidden_states,
                "attentions": all_self_attns,
                "causal_mask": causal_mask,
                "seq_len": L
            }
        else:
            out = (hidden_states, next_cache, all_hidden_states, all_self_attns, causal_mask, L)
            return out[:3] + (causal_mask, L)


class LlamaStage2(nn.Module):
    """
    Stage2：后续 server_layers 个 LlamaDecoderLayer + RMSNorm
    """
    def __init__(self, layers: nn.ModuleList, rms_norm: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm   = rms_norm

    def forward(self,
                hidden_states: torch.FloatTensor,  # [B, L, H] 来自 Stage1
                causal_mask: torch.Tensor,         # [B, 1, L, L]，dtype=torch.bool
                seq_len: int,                      # Stage1 当前的序列长度
                past_key_values=None,
                output_attentions: bool = False,
                use_cache: bool = False
               ):
        all_hidden_states = () if output_attentions or use_cache else None
        all_self_attns   = () if output_attentions else None
        next_cache       = None

        # 1) 逐层过后 server_layers 层；**不传 position_ids, cache_position**
        for layer in self.layers:
            if output_attentions:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,   # 布尔型 causal_mask
                output_attentions=output_attentions,
                use_cache=use_cache
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_cache = layer_outputs[1] if not output_attentions else layer_outputs[2]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # 2) 最后一层做 RMSNorm
        hidden_states = self.norm(hidden_states)  # [B, L, H]

        if use_cache or output_attentions:
            if output_attentions:
                return hidden_states, next_cache, all_hidden_states, all_self_attns
            return hidden_states, next_cache
        return hidden_states

# ——————————————————————————————————————————————————————————————————————————————————
# 4. 构建 Stage1/Stage2 子模块，并加载拆分后的 safetensors 权重
# ——————————————————————————————————————————————————————————————————————————————————
print("3️⃣ 构建 Stage1/Stage2 子模块，并加载拆分权重……")
# 4.1 实例化子模块（embed_tokens, layers 已经从 full_model 拆出）
stage1_net = LlamaStage1(embed_tokens=embed_tokens, layers=stage1_layers)
stage2_net = LlamaStage2(layers=stage2_layers, rms_norm=rms_norm)

# 4.2 lm_head 也已从 full_model 拆出，后面一并转到 GPU

# 4.3 加载 model_stage1.safetensors
print("   → 载入 model_stage1.safetensors ……")
state_stage1 = load_file(stage1_path)
client_state_dict = {}
for full_key, tensor in state_stage1.items():
    # safetensors 的 key 通常带前缀 "model."
    new_key = full_key.removeprefix("model.")
    client_state_dict[new_key] = tensor
# 把前 12 层 + embed_tokens 的权重 load 进 stage1_net
stage1_net.load_state_dict(client_state_dict, strict=False)

# 4.4 加载 model_stage2.safetensors
print("   → 载入 model_stage2.safetensors ……")
state_stage2 = load_file(stage2_path)
server_state_dict = {}
for full_key, tensor in state_stage2.items():
    new_key = full_key.removeprefix("model.")
    if new_key == "lm_head.weight":
        # lm_head.weight 单独赋给 lm_head（Float32 再转换到 FP16）
        lm_head.weight.data = tensor.to(torch.float32)
    else:
        # new_key 形式如 "layers.12.self_attn.q_proj.weight"
        if new_key.startswith("layers."):
            parts = new_key.split(".")
            layer_idx = int(parts[1])               # 12..31
            new_idx = layer_idx - client_layers      # 12→0, 13→1, …, 31→19
            mapped_key = new_key.replace(f"layers.{layer_idx}", f"layers.{new_idx}")
            server_state_dict[mapped_key] = tensor
        else:
            # norm.weight / norm.bias
            server_state_dict[new_key] = tensor
# 把后 20 层 + RMSNorm 的权重加载到 stage2_net
stage2_net.load_state_dict(server_state_dict, strict=False)

# 4.5 将所有子模块转成 FP16 并移到 GPU
device = "cuda:0"
stage1_net = stage1_net.half().to(device)
stage2_net = stage2_net.half().to(device)
lm_head    = lm_head.half().to(device)
print("   → Stage1/Stage2/LMHead 已转为 FP16 并移至 GPU。")

# 4.6 切换为评估模式
stage1_net.eval()
stage2_net.eval()

# ——————————————————————————————————————————————————————————————————————————————————
# 5. 定义 Split 推理函数：generate_split()
#
#    思路：
#      1) 把 prompt → input_ids [1, L], attention_mask [1, L]
#      2) 在每一步循环里：
#         a. Stage1: 传 (input_ids, attention_mask)，不传 position_ids，让官方子模块自动生成 1D pos_ids，
#            同时拿到 bool 型的 causal_mask1 以及中间 hidden1。
#         b. Stage2: 传 (hidden1, causal_mask1)，不传 position_ids，让官方子模块自动维持序列长度。
#         c. hidden2 → lm_head → logits[:, -1] → argmax 得到下一个 token_id。
#         d. 把下一个 token_id 拼到 input_ids/attention_mask，继续循环。
# ——————————————————————————————————————————————————————————————————————————————————
def generate_split(stage1, stage2, head, tokenizer, prompt: str, max_new_tokens: int = 100):
    """
    Args:
      stage1:    LlamaStage1 子模块
      stage2:    LlamaStage2 子模块
      head:      对应 lm_head（nn.Linear）
      tokenizer: HF AutoTokenizer
      prompt:    初始字符串
      max_new_tokens: 最多生成的 token 数
    Returns:
      完整的生成：prompt + 模型生成内容
    """
    device = next(stage1.parameters()).device

    # 1) 把 prompt 编码成 input_ids [1, L] & attention_mask [1, L]
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids      = inputs["input_ids"].to(device)       # [1, L]
    attention_mask = inputs["attention_mask"].to(device)  # [1, L]

    generated = prompt
    for step in range(max_new_tokens):
        print(f"step: {step} running...")
        B, L = input_ids.shape

        with torch.no_grad():
            # 2.a) 送给 Stage1，不传 position_ids
            out1 = stage1(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False
            )
            hidden1      = out1["last_hidden_state"]  # [1, L, H]
            causal_mask1 = out1["causal_mask"]        # [1, 1, L, L], dtype=torch.bool
            seq_len1     = out1["seq_len"]            # 等于 L

            # 2.b) 送给 Stage2，不传 position_ids
            hidden2 = stage2(
                hidden_states=hidden1,
                causal_mask=causal_mask1,
                seq_len=seq_len1,
                use_cache=False,
                output_attentions=False
            )  # → [1, L, H]

            # 2.c) hidden2 → lm_head → 获取 logits 最后一位 → argmax 得到 next_token_id
            logits = head(hidden2)                       # [1, L, vocab_size]
            next_token_logits = logits[:, -1, :]          # [1, vocab_size]
            next_token_id     = torch.argmax(next_token_logits, dim=-1)  # [1]
            next_id = next_token_id.item()

        # 2.d) decode 为文本，拼接到 generated
        new_tok_str = tokenizer.decode(next_id)
        generated   += new_tok_str

        # 2.e) 更新 input_ids 和 attention_mask
        next_id_tensor = next_token_id.unsqueeze(0)                   # [1,1]
        input_ids      = torch.cat([input_ids, next_id_tensor], dim=1)  # [1, L+1]
        new_attn       = torch.ones((B, 1), dtype=attention_mask.dtype, device=device)
        attention_mask = torch.cat([attention_mask, new_attn], dim=1)     # [1, L+1]

        # 如果遇到 EOS，就退出
        if next_id == tokenizer.eos_token_id:
            break

    return generated

# ——————————————————————————————————————————————————————————————————————————————————
# 6. 测试推理
# ——————————————————————————————————————————————————————————————————————————————————
if __name__ == "__main__":
    prompt_text = "Who is Crayon Shinchan?\n"
    print("🚀 开始 Split 推理……")
    output_text = generate_split(stage1_net, stage2_net, lm_head, tokenizer, prompt_text, max_new_tokens=150)
    print("最终生成：")
    print(output_text)
