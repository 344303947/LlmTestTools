#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# MIT License - Copyright (c) 2025 LlmTestTools
# https://github.com/344303947/LlmTestTools

import requests
import time
import sys
import json
import random
import tiktoken
import io
from datetime import datetime
from typing import List, Dict, Optional, Generator

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

API_URL = "http://192.168.0.32:9070/v1/chat/completions"
API_KEY = "sk_344303"
MODEL_NAME = "Qwen3.5-122B-A10B"
INPUT_MIN_LEN_DEFAULT = 128
INPUT_MAX_LEN_DEFAULT = 4096
OUTPUT_LEN = 128
CONTEXT_LEN = 16
CONNECT_TIMEOUT = 180
READ_TIMEOUT = 200

FIXED_PROMPT = "\nRepeat the above content one hundred times."
WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

try:
    enc = tiktoken.encoding_for_model("gpt-4")
except KeyError:
    enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def generate_user_prompt(length: int) -> str:
    fixed_tokens = count_tokens(FIXED_PROMPT)
    need_word_length = max(0, length - fixed_tokens)

    user_prompt_parts = []
    for i in range(need_word_length):
        word = random.choice(WORDS)
        if i > 0:
            user_prompt_parts.append(" ")
        user_prompt_parts.append(word)

    user_prompt_parts.append(FIXED_PROMPT)
    return "".join(user_prompt_parts)


def stream_response(prompt: str, max_tokens: int) -> Generator[Dict, None, None]:
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "stream": True,
    }

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    try:
        response = requests.post(
            API_URL,
            json=payload,
            headers=headers,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            stream=True,
        )

        if response.status_code != 200:
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {})
                if isinstance(error_msg, dict):
                    error_detail = error_msg.get("message", response.text)
                else:
                    error_detail = str(error_msg)
            except (json.JSONDecodeError, ValueError):
                error_detail = response.text
            raise Exception(f"HTTP {response.status_code}: {error_detail}")

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("data:"):
                    json_str = decoded_line[5:].strip()
                    if json_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(json_str)
                        yield chunk
                    except json.JSONDecodeError:
                        continue

    except requests.exceptions.Timeout:
        raise Exception("Request timeout (connect or read)")
    except requests.exceptions.ConnectionError:
        raise Exception("Connection error")
    except Exception as e:
        raise Exception(f"Stream error: {str(e)}")


def test_stream_prompt(input_len: int, output_len: int) -> Optional[Dict]:
    try:
        prompt = generate_user_prompt(input_len)
    except Exception as e:
        print(f"[ERROR] 生成 prompt 失败：{str(e)}")
        return None

    try:
        request_start = time.time()
        first_token_time: float | None = None
        last_token_time: float | None = None
        token_count = 0

        generated_content = ""
        for chunk in stream_response(prompt, output_len):
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content") or delta.get("reasoning")
                if content:
                    generated_content += content
                    if first_token_time is None:
                        first_token_time = time.time()
                    last_token_time = time.time()

        if generated_content:
            token_count = len(enc.encode(generated_content))

        if token_count == 0 or first_token_time is None or last_token_time is None:
            raise Exception("No tokens received")

        ttft_ms = (first_token_time - request_start) * 1000
        decode_duration_ms = (last_token_time - first_token_time) * 1000
        total_duration_ms = (last_token_time - request_start) * 1000

        prefill_speed = input_len / (ttft_ms / 1000) if ttft_ms > 0 else 0
        decode_duration_sec = (
            max(decode_duration_ms / 1000, 0.01) if decode_duration_ms > 0 else 0.01
        )
        decode_speed = token_count / decode_duration_sec

        return {
            "input_len": input_len,
            "prefill_ms": round(ttft_ms, 2),
            "prefill_speed": round(prefill_speed, 1),
            "output_len": token_count,
            "decode_ms": round(decode_duration_ms, 2),
            "decode_speed": round(decode_speed, 1),
            "total_ms": round(total_duration_ms, 2),
            "prompt_tokens": count_tokens(prompt),
            "fixed_prompt_tokens": count_tokens(FIXED_PROMPT),
            "generated_words": max(0, input_len - count_tokens(FIXED_PROMPT)),
        }

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return None


def generate_test_points(min_len: int, max_len: int) -> List[int]:
    points = []
    current = min_len
    while current <= max_len:
        points.append(current)
        current *= 2
    return points


def main():
    global MODEL_NAME

    print("\n=== LLM API 性能测试 ===")
    print(f"接口：{API_URL}")
    print(f"输出长度：{OUTPUT_LEN} tokens")
    print("=" * 80)
    print(f"🔗 项目地址：https://github.com/344303947/LlmTestTools")
    print("=" * 80)

    INPUT_MIN_LEN = INPUT_MIN_LEN_DEFAULT
    INPUT_MAX_LEN = INPUT_MAX_LEN_DEFAULT
    MODEL_NAME = "Qwen3.5-122B-A10B"

    if len(sys.argv) >= 4:
        try:
            INPUT_MIN_LEN = int(sys.argv[1])
            INPUT_MAX_LEN = int(sys.argv[2])
            MODEL_NAME = sys.argv[3]
            if INPUT_MIN_LEN < 1:
                raise ValueError("输入长度必须 ≥ 1")
            if INPUT_MAX_LEN < INPUT_MIN_LEN:
                raise ValueError("最大长度不能小于最小长度")
            print(f"输入范围：[{INPUT_MIN_LEN}, {INPUT_MAX_LEN}] tokens (命令行参数)")
            print(f"模型：{MODEL_NAME} (命令行参数)")
        except ValueError as e:
            print(f"参数错误：{e}，使用默认值")
            print(f"模型：{MODEL_NAME} (默认)")
    elif len(sys.argv) == 3:
        try:
            INPUT_MIN_LEN = int(sys.argv[1])
            INPUT_MAX_LEN = int(sys.argv[2])
            if INPUT_MIN_LEN < 1:
                raise ValueError("输入长度必须 ≥ 1")
            if INPUT_MAX_LEN < INPUT_MIN_LEN:
                raise ValueError("最大长度不能小于最小长度")
            print(f"输入范围：[{INPUT_MIN_LEN}, {INPUT_MAX_LEN}] tokens (命令行参数)")
            print(f"模型：{MODEL_NAME} (默认)")
        except ValueError as e:
            print(f"参数错误：{e}，使用默认值")
            print(f"模型：{MODEL_NAME} (默认)")
    else:
        print(f"输入范围：[{INPUT_MIN_LEN}, {INPUT_MAX_LEN}] tokens (默认)")
        print(f"模型：{MODEL_NAME} (默认)")

    test_points = generate_test_points(INPUT_MIN_LEN, INPUT_MAX_LEN)
    print(f"测试点：{test_points}")
    print("=" * 80)

    headers = ["输入长度", "预填充(t/s)", "输出(t/s)", "TTFT(ms)", "输出延迟(ms)"]
    widths = [10, 14, 12, 12, 14]

    header_line = "  ".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    print(f"\n{header_line}")
    print("-" * len(header_line))

    results = []
    total_tests = len(test_points)

    for idx, input_len in enumerate(test_points, 1):
        print(f"[{idx}/{total_tests}] 测试 {input_len} tokens", end=" ", flush=True)
        result = test_stream_prompt(input_len, OUTPUT_LEN)

        if result:
            print("OK")
            row = [
                result["input_len"],
                result["prefill_speed"],
                result["decode_speed"],
                result["prefill_ms"],
                result["decode_ms"],
            ]
            formatted_row = "  ".join(f"{v:<{w}}" for v, w in zip(row, widths))
            print(f"        {formatted_row}")
            results.append(result)
        else:
            print("FAILED")
            if len(results) == 0 and idx == 1:
                print(f"  ⚠️  建议检查模型名称 '{MODEL_NAME}' 是否正确")

    print("-" * 80)
    print(f"测试完成：{len(results)}/{total_tests} 成功")

    if results:
        avg_ttft = sum(r["prefill_ms"] for r in results) / len(results)
        avg_decode_rate = sum(r["decode_speed"] for r in results) / len(results)

        print(f"\n统计摘要:")
        print(f"  平均 TTFT: {avg_ttft:.2f} ms")
        print(f"  平均输出速度：{avg_decode_rate:.2f} t/s")

    print("=" * 80)
    print(f"🔗 项目地址：https://github.com/344303947/LlmTestTools")
    print(f"⭐ 如果有帮助，请给个 Star!")
    print("=" * 80)


if __name__ == "__main__":
    main()
