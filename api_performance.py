#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import time
import sys
import json
import random
import tiktoken
from datetime import datetime
from typing import List, Dict, Optional, Generator

# ==================== 配置区 ====================
API_URL = "http://192.168.0.32:9070/v1/chat/completions"
API_KEY = "sk_344303"
MODEL_NAME = "Qwen3-Next-80B-A3B-Instruct"
INPUT_MIN_LEN_DEFAULT = 128
INPUT_MAX_LEN_DEFAULT = 4096
OUTPUT_LEN = 128
CONTEXT_LEN = 16  # 模型上下文长度，仅作说明
CONNECT_TIMEOUT = 180  # 连接超时（秒）
READ_TIMEOUT = 200  # 读取超时（秒）

# ==================== 提示词与词表 ====================
FIXED_PROMPT = "\nRepeat the above content one hundred times."
WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

# ==================== Tokenizer ====================
try:
    enc = tiktoken.encoding_for_model("gpt-4")
except KeyError:
    enc = tiktoken.get_encoding("cl100k_base")  # fallback


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
            raise Exception(f"HTTP {response.status_code}: {response.text}")

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
        print(f"[ERROR] 生成 prompt 失败: {str(e)}")
        return None

    try:
        request_start = time.time()
        first_token_time = None
        last_token_time = None
        token_count = 0

        for chunk in stream_response(prompt, output_len):
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta and delta["content"] is not None:
                    token_count += 1

                    if first_token_time is None:
                        first_token_time = time.time()
                    last_token_time = time.time()

        if token_count == 0:
            return None

        ttft_ms = (first_token_time - request_start) * 1000
        decode_duration_ms = (last_token_time - first_token_time) * 1000
        total_duration_ms = (last_token_time - request_start) * 1000

        prefill_speed = input_len / (ttft_ms / 1000) if ttft_ms > 0 else 0
        decode_speed = (
            token_count / (decode_duration_ms / 1000) if decode_duration_ms > 0 else 0
        )

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
    print("\n=== LLM API 性能测试 ===")
    print(f"接口：{API_URL}")
    print(f"模型：{MODEL_NAME}")
    print(f"输出长度：{OUTPUT_LEN} tokens")
    print("=" * 80)
    print(f"🔗 项目地址：https://github.com/344303947/LlmTestTools")
    print(f"⭐ 如果有帮助，请给个 Star!")
    print("=" * 80)

    # ========== 命令行参数 ==========
    if len(sys.argv) >= 3:
        try:
            INPUT_MIN_LEN = int(sys.argv[1])
            INPUT_MAX_LEN = int(sys.argv[2])
            if INPUT_MIN_LEN < 1:
                raise ValueError("输入长度必须 ≥ 1")
            if INPUT_MAX_LEN < INPUT_MIN_LEN:
                raise ValueError("最大长度不能小于最小长度")
            print(f"输入范围: [{INPUT_MIN_LEN}, {INPUT_MAX_LEN}] tokens (命令行参数)")
        except ValueError as e:
            print(f"参数错误: {e}，使用默认值")
            INPUT_MIN_LEN = INPUT_MIN_LEN_DEFAULT
            INPUT_MAX_LEN = INPUT_MAX_LEN_DEFAULT
    else:
        INPUT_MIN_LEN = INPUT_MIN_LEN_DEFAULT
        INPUT_MAX_LEN = INPUT_MAX_LEN_DEFAULT
        print(f"输入范围: [{INPUT_MIN_LEN}, {INPUT_MAX_LEN}] tokens (默认)")

    # 生成测试点
    test_points = generate_test_points(INPUT_MIN_LEN, INPUT_MAX_LEN)
    print(f"测试点: {test_points}")
    print("=" * 80)

    # ========== 表头 ==========
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

    print("-" * 80)
    print(f"测试完成: {len(results)}/{total_tests} 成功")

    # ========== 统计摘要 ==========
    if results:
        avg_ttft = sum(r["prefill_ms"] for r in results) / len(results)
        avg_decode_rate = sum(r["decode_speed"] for r in results) / len(results)

        print(f"\n统计摘要:")
        print(f"  平均 TTFT: {avg_ttft:.2f} ms")
        print(f"  平均输出速度: {avg_decode_rate:.2f} t/s")

        # ========== CSV 导出 ==========
        try:
            import csv

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"api_performance_{timestamp}.csv"

            fieldnames = [
                "InputLen",
                "TTFT_ms",
                "PrefillRate_tok_s",
                "OutputLen",
                "TPOT_ms",
                "DecodeRate_tok_s",
                "Total_ms",
                "PromptTokens",
                "FixedPromptTokens",
                "GeneratedWords",
            ]

            # with open(csv_filename, "w", newline="", encoding="utf-8") as f:
            #     writer = csv.DictWriter(f, fieldnames=fieldnames)
            #     writer.writeheader()

            #     for r in results:
            #         writer.writerow({
            #             "InputLen": r["input_len"],
            #             "TTFT_ms": r["prefill_ms"],
            #             "PrefillRate_tok_s": r["prefill_speed"],
            #             "OutputLen": r["output_len"],
            #             "TPOT_ms": r["decode_ms"],
            #             "DecodeRate_tok_s": r["decode_speed"],
            #             "Total_ms": r["total_ms"],
            #             "PromptTokens": r["prompt_tokens"],
            #             "FixedPromptTokens": r["fixed_prompt_tokens"],
            #             "GeneratedWords": r["generated_words"]
            #         })

            # print(f"\n结果已保存: {csv_filename}")
        except Exception as e:
            print(f"CSV 保存失败: {e}")


if __name__ == "__main__":
    main()
