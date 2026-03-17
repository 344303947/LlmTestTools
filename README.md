# LLM API Performance Test Tool

LLM API 性能测试工具，用于测试大语言模型的推理性能指标。

## 功能

- 测试不同输入长度下的预填充速度 (prefill speed)
- 测试输出速度 (decode speed / tokens per second)
- 测量首 token 时间 (TTFT - Time To First Token)
- 支持自定义输入范围和模型名称

## 安装依赖

```bash
pip install requests tiktoken
```

## 使用方法

```bash
python api_performance.py [input_min] [input_max] [model_name]
```

### 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| input_min | 可选 | 128 | 最小输入 token 数 |
| input_max | 可选 | 4096 | 最大输入 token 数 |
| model_name | 可选 | Qwen3.5-122B-A10B | 要测试的模型名称 |

### 示例

```bash
# 使用默认参数测试
python api_performance.py

# 指定输入范围和模型
python api_performance.py 128 64000 Qwen3.5-122B-A10B
```

## 输出指标

| 指标 | 说明 |
|------|------|
| 输入长度 | 测试的输入 token 数 |
| 预填充 (t/s) | 输入处理速度 (tokens/second) |
| 输出 (t/s) | 生成速度 (tokens/second) |
| TTFT (ms) | 首 token 时间 (毫秒) |
| 输出延迟 (ms) | 完整输出耗时 (毫秒) |

## 注意事项

1. 确保 API 服务正常运行
2. 检查 API 密钥是否正确
3. 根据模型实际能力调整测试范围

## License

Copyright (c) 2024 LlmTestTools
