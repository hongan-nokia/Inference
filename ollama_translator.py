# -*- coding: utf-8 -*-
"""
@Time: 3/3/2025 11:43 AM
@Author: Honggang Yuan
@Email: honggang.yuan@nokia-sbell.com
Description: 
"""
import requests
import json
import langdetect


def translate_with_ollama(text):
    detected_lang = langdetect.detect(text)
    if detected_lang.startswith('zh'):
        prompt = f"Translate the following Chinese text into English: {text}"
    else:
        prompt = f"将以下英文翻译成中文：{text}"
    data = {
        "model": "DeepSeek-R1-Q2_K",
        "prompt": prompt,
        "stream": False,
    }
    response = requests.post("http://135.251.50.42:11434/api/generate", json=data)
    print(response)
    if response.status_code == 200:
        print(response.text)
        return f"请求成功: {response.text}"
    else:
        return f"请求失败: {response.status_code}"


if __name__ == "__main__":
    # user_input = "八百标兵奔北坡"
    user_input = "Resource Provision is Important in Microservice Application"
    translation = translate_with_ollama(user_input)
    print("翻译结果:", translation)
