import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
from safetensors.torch import load_file
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


def load_stage_from_file(model, stage_path, stage_layer_indices, device):
    weights = load_file(stage_path)
    layers = []
    for i in stage_layer_indices:
        layer = model.model.layers[i].to(device)
        prefix = f"model.layers.{i}."
        with torch.no_grad():
            for name, param in layer.named_parameters():
                full_name = prefix + name
                if full_name in weights:
                    param.copy_(weights[full_name].to(device))
                else:
                    print(f"Warning: {full_name} not found in {stage_path}")
        layers.append(layer)
    return layers


def build_position_embeddings(config, seq_len, device):
    """
    构造一次长度为 seq_len 的 (cos, sin)，后续按需切片复用。
    """
    head_dim = config.hidden_size // config.num_attention_heads
    rope = LlamaRotaryEmbedding(config, device=device)
    dummy_x = torch.zeros(1, seq_len, head_dim, device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    cos, sin = rope(dummy_x, position_ids)  # [1, seq_len, head_dim]
    return cos, sin


def run_manual_forward(layer, hidden_states, attention_mask, config, cos, sin, past_key_value=None):
    """
    单层前向，用切片后的 cos/sin 做 Rotary Embedding。
    如果 layer(...) 返回长度1的 tuple，仅包含 hidden_states；否则解包前两个元素。
    """
    seq_len = hidden_states.size(1)
    # 构造 position_ids = [0..seq_len-1]
    position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(hidden_states.size(0), seq_len)
    # 取对应长度的 cos/sin 片段
    cos_slice = cos[:, :seq_len]
    sin_slice = sin[:, :seq_len]
    # attention_mask → [B,1,1,seq_len]
    causal_mask = attention_mask.bool().unsqueeze(1).unsqueeze(1)  # [B,1,1,seq_len]

    with torch.no_grad():
        output = layer(
            hidden_states=hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=(cos_slice, sin_slice),
        )

    # 调试：打印 output 类型和长度
    # print(f">>> Debug: layer returned type {type(output)}")
    # if isinstance(output, tuple):
    #     print(f">>> Debug: tuple length = {len(output)}")
    #     for idx, item in enumerate(output):
    #         if isinstance(item, torch.Tensor):
    #             print(f"  - output[{idx}] tensor shape: {item.shape}")
    #         else:
    #             print(f"  - output[{idx}] type: {type(item)}")
    # elif hasattr(output, "last_hidden_state"):
    #     print(f">>> Debug: ModelOutput with keys {list(output.to_dict().keys()) if hasattr(output, 'to_dict') else output.keys()}")
    # else:
    #     print(f">>> Debug: single Tensor, shape = {output.shape if isinstance(output, torch.Tensor) else type(output)}")

    # 安全解包
    if isinstance(output, tuple):
        if len(output) == 1:
            hidden_states = output[0]
            past = None
        else:
            hidden_states = output[0]
            past = output[1]
    elif hasattr(output, "last_hidden_state"):
        hidden_states = output.last_hidden_state
        past = None
    else:
        hidden_states = output
        past = None

    return hidden_states, past


def run_stage_forward(layers, hidden_states, attention_mask, config, cos, sin):
    """
    对一组层（stage）依次前向；
    每层都不使用缓存 past_key_value，返回 None 占位。
    """
    new_past = []
    for layer in layers:
        hidden_states, _ = run_manual_forward(
            layer,
            hidden_states,
            attention_mask,
            config,
            cos=cos,
            sin=sin,
            past_key_value=None
        )
        new_past.append(None)
    return hidden_states, new_past


def sample_next_token(logits, temperature=1.0, top_k=50):
    """
    top-k + temperature 采样：输入 logits 形状 [B, vocab_size]，输出 [B,1] 下一个 token id
    """
    logits = logits / temperature
    topk_logits, topk_indices = torch.topk(logits, k=top_k, dim=-1)
    probs = torch.softmax(topk_logits, dim=-1)
    choice = torch.multinomial(probs, num_samples=1)
    return topk_indices.gather(-1, choice)


def full_inference(base_model, stage1_path, stage2_path, input_text, device,
                   max_new_tokens=20, temperature=1.0, top_k=50):
    config = LlamaConfig.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="cpu", trust_remote_code=True)
    model.eval()

    # 1) 准备 input_ids
    input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"].to(device)
    prompt_len = input_ids.size(1)

    # 2) 把 embed_tokens、lm_head、以及最后的 norm 移到 GPU
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model.lm_head = model.lm_head.to(device)
    model.model.norm = model.model.norm.to(device)

    # 3) 加载拆分权重到对应层
    stage1_layers = load_stage_from_file(model, stage1_path, list(range(0, 12)), device)
    stage2_layers = load_stage_from_file(model, stage2_path, list(range(12, 32)), device)

    # 4) 构造 rotary (cos, sin)，长度 = prompt_len + max_new_tokens
    max_seq_len = prompt_len + max_new_tokens
    cos, sin = build_position_embeddings(config, max_seq_len, device)

    # 5) 逐 token 生成（每步 embed 整个已生成序列）
    generated = input_ids.clone()
    for step in range(max_new_tokens):
        print(f"➡️ Step {step} running...")
        # 5.1) 对当前生成的整个序列进行 embedding
        hidden_states = model.model.embed_tokens(generated)  # [1, cur_len, H]
        cur_len = hidden_states.size(1)

        # 5.2) 构造 attention_mask = 全 1，长度为 cur_len
        attention_mask = torch.ones((1, cur_len), device=device)

        # 5.3) 运行前 12 层
        hidden_states, _ = run_stage_forward(
            stage1_layers,
            hidden_states,
            attention_mask,
            config,
            cos,
            sin
        )
        # 5.4) 运行后 20 层
        hidden_states, _ = run_stage_forward(
            stage2_layers,
            hidden_states,
            attention_mask,
            config,
            cos,
            sin
        )

        # 5.5) 跑完所有 32 层后做一次 LayerNorm
        hidden_states = model.model.norm(hidden_states)  # [1, cur_len, H]

        # 5.6) lm_head → logits → 采样下一个 token
        last_hidden = hidden_states[:, -1, :]  # [1, H]
        with torch.no_grad():
            logits = model.lm_head(last_hidden)  # [1, vocab_size]
            next_token = sample_next_token(logits, temperature, top_k)  # [1,1]

        # 5.7) 拼接 generated，并检查 EOS
        generated = torch.cat([generated, next_token], dim=1)  # [1, cur_len+1]
        if next_token.item() == tokenizer.eos_token_id:
            break

    # 6) 解码 prompt 后新增的部分
    new_tokens = generated[0, prompt_len:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return output_text


if __name__ == "__main__":
    # base_model    = "/guozhanqiu/LoRA-FT/Fine-tuned/merge_lora_math-f12"
    base_model = "/guozhanqiu/LLModels/llama3_8B_Base"
    stage1_path = "/guozhanqiu/model_layers/model_stage1.safetensors"
    stage2_path = "/guozhanqiu/model_layers/model_stage2.safetensors"
    input_text = "What is the capital of France?"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output = full_inference(base_model, stage1_path, stage2_path, input_text, device, max_new_tokens=100, temperature=1, top_k=50)
    print("✅ Input:", input_text)
    print("✅ Output:", output)
