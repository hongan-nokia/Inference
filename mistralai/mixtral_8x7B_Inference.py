# -*- coding: utf-8 -*-
"""
@Time: 1/13/2025 8:19 PM
@Author: Honggang Yuan
@Email: honggang.yuan@nokia-sbell.com
Description: 
"""
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

model = Transformer.from_folder(mistral_models_path)
out_tokens, _ = generate([tokens], model, max_tokens=64, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)

result = tokenizer.decode(out_tokens[0])

print(result)
