# 纯CPU运行 极简版 test_qwen.py
import os
import torch  # 必须导入

# 配置国内镜像+禁用GPU
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from transformers import pipeline

# 加载模型：只保留CPU和低内存参数，去掉所有量化配置
pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2-0.5B-Instruct",  # 如果你是本地模型，换成本地路径
    model_kwargs={
        "device_map": "cpu",          # 强制CPU
        "low_cpu_mem_usage": True     # 优化CPU内存占用
    }
)

# 测试生成（缩短生成长度，进一步降内存）
prompt = "请用一段话描述石化生产的标准流程"
# output = pipe(
#     prompt,
#     max_new_tokens=100  # 生成长度设小一点，避免内存不足
#     do_sample=False     # 关闭采样，降内存
#
# )
output = pipe(
    prompt,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True  # 关键：开启采样，生成自然文本
)
print("模型回答：\n", output[0]['generated_text'])