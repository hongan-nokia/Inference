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


def build_position_embeddings(layer, config, seq_len, device):
    rope = LlamaRotaryEmbedding(config, device=device)
    dummy_x = torch.zeros(1, seq_len, layer.self_attn.head_dim, device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    cos, sin = rope(dummy_x, position_ids)
    return (cos, sin)


def run_manual_forward(layer, hidden_states, attention_mask, config, past_key_value=None, position_ids=None):
    input_shape = attention_mask.shape
    seq_len = input_shape[-1]
    if position_ids is None:
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(input_shape)
    position_embeddings = build_position_embeddings(layer, config, seq_len, hidden_states.device)

    if attention_mask.dim() == 2:
        attention_mask = attention_mask[:, None, None, :]
    attention_mask = attention_mask.to(dtype=hidden_states.dtype)

    with torch.no_grad():
        output = layer.forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=False,
            use_cache=True,
            cache_position=None,
            position_embeddings=position_embeddings,
        )

    if isinstance(output, tuple):
        hidden_states = output[0]
        past = output[1] if len(output) > 1 else None
        return hidden_states, past
    return output, None


def run_stage_forward(layers, hidden_states, attention_mask, config, past_key_values, offset, step, device):
    new_past = []
    position_ids = torch.tensor([[offset + step]], device=device)
    for i, layer in enumerate(layers):
        hidden_states, past = run_manual_forward(
            layer, hidden_states, attention_mask, config,
            past_key_value=past_key_values[offset + i],
            position_ids=position_ids
        )
        new_past.append(past)
    return hidden_states, new_past


def sample_next_token(logits, temperature=1.0, top_k=50):
    logits = logits / temperature
    topk_logits, topk_indices = torch.topk(logits, k=top_k, dim=-1)
    probs = torch.softmax(topk_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return topk_indices.gather(-1, next_token)


def full_inference(base_model, stage1_path, stage2_path, input_text, device,
                   max_new_tokens=20, temperature=1.0, top_k=50):
    config = LlamaConfig.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="cpu", trust_remote_code=True)
    model.eval()

    input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"].to(device)
    original_len = input_ids.shape[1]

    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model.lm_head = model.lm_head.to(device)
    model.model.norm = model.model.norm.to(device)

    past_key_values = [None] * model.config.num_hidden_layers

    # 前 12 层
    stage1_layers = load_stage_from_file(model, stage1_path, list(range(0, 12)), device)
    # 后 20 层
    stage2_layers = load_stage_from_file(model, stage2_path, list(range(12, 32)), device)

    for step in range(max_new_tokens):
        print(f"➡️ Step {step} running...")
        input_token = input_ids if step == 0 else next_token
        attention_mask = torch.ones_like(input_token)
        hidden_states = model.model.embed_tokens(input_token)

        # 前 12 层
        hidden_states, past1 = run_stage_forward(
            stage1_layers, hidden_states, attention_mask, config,
            past_key_values, offset=0, step=step, device=device
        )
        # hidden_states = model.model.norm(hidden_states)

        # 后 20 层
        hidden_states, past2 = run_stage_forward(
            stage2_layers, hidden_states, attention_mask, config,
            past_key_values, offset=12, step=step, device=device
        )

        past_key_values = past1 + past2
        # Apply final LayerNorm before lm_head
        hidden_states = model.model.norm(hidden_states)  # [1, cur_len, H]

        # Compute logits and sample next token
        last_hidden = hidden_states[:, -1]
        with torch.no_grad():
            logits = model.lm_head(last_hidden)
            next_token = sample_next_token(logits, temperature, top_k)

        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    new_tokens = input_ids[0, original_len:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return output_text


if __name__ == "__main__":
    # base_model    = "/guozhanqiu/LoRA-FT/Fine-tuned/merge_lora_math-f12"
    base_model = "/guozhanqiu/LLModels/llama3_8B_Base"
    stage1_path = "/guozhanqiu/model_layers/model_stage1.safetensors"
    stage2_path = "/guozhanqiu/model_layers/model_stage2.safetensors"
    input_text = "The capital of USA is"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output = full_inference(
        base_model, stage1_path, stage2_path, input_text, device,
        max_new_tokens=100, temperature=0.7, top_k=50
    )
    print("✅ Input:", input_text)
    print("✅ Output:", output)
