# LLM API Performance Test Tool

LLM API 性能测试工具，用于测试大语言模型的推理性能指标。

## 功能

- 测试不同输入长度下的预填充速度 (prefill speed)
- 测试输出速度 (decode speed / tokens per second)
- 测量首 token 时间 (TTFT - Time To First Token)
- 支持自定义输入范围和模型名称
- 自动检测模型是否存在

## 安装依赖

```bash
pip install requests tiktoken
```

## 使用方法

### 基本用法

```bash
python api_performance.py [input_min] [input_max] [model_name]
```

### 直接执行

脚本已设置执行权限，可直接运行：

```bash
./api_performance.py [input_min] [input_max] [model_name]
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

# 只指定输入范围，使用默认模型
python api_performance.py 128 1024

# 使用不同模型测试
python api_performance.py 128 512 Qwen2.5-7B
```

## 输出说明

测试会输出以下指标：

| 指标 | 说明 |
|------|------|
| 输入长度 | 测试的输入 token 数 |
| 预填充 (t/s) | 输入处理速度 (tokens/second) |
| 输出 (t/s) | 生成速度 (tokens/second) |
| TTFT (ms) | 首 token 时间 (毫秒) |
| 输出延迟 (ms) | 完整输出耗时 (毫秒) |

### 示例输出

```
=== LLM API 性能测试 ===
接口：http://192.168.0.32:9070/v1/chat/completions
输出长度：128 tokens
================================================================================
输入范围：[128, 512] tokens (命令行参数)
模型：Qwen3.5-122B-A10B (命令行参数)
测试点：[128, 256, 512]
================================================================================

输入长度        预填充 (t/s)        输出 (t/s)       TTFT(ms)      输出延迟 (ms)      
----------------------------------------------------------------------
[1/3] 测试 128 tokens OK
        128         106.2           28.7          1205.24       4466.28       
[2/3] 测试 256 tokens OK
        256         123.6           28.6          2071.98       4469.88       
[3/3] 测试 512 tokens OK
        512         221.1           28.7          2315.87       4465.26       
--------------------------------------------------------------------------------
测试完成：3/3 成功

统计摘要:
  平均 TTFT: 1864.36 ms
  平均输出速度：28.67 t/s
```

## 错误处理

- **模型不存在**: 显示 HTTP 404 错误并提示检查模型名称
- **连接超时**: 显示连接错误信息
- **请求失败**: 显示具体的错误原因

## 配置

在脚本顶部可以修改以下配置：

```python
API_URL = "http://192.168.0.32:9070/v1/chat/completions"  # API 地址
API_KEY = "sk_344303"                                      # API 密钥
OUTPUT_LEN = 128                                           # 输出 token 数
CONNECT_TIMEOUT = 180                                      # 连接超时 (秒)
READ_TIMEOUT = 200                                         # 读取超时 (秒)
```

## 注意事项

1. 确保 API 服务正常运行
2. 检查 API 密钥是否正确
3. 根据模型实际能力调整测试范围
4. 输出速度受多种因素影响，建议多次测试取平均值

## License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) 2025 LlmTestTools
