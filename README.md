# 石化行业本地RAG智能问答系统
轻量型本地部署的RAG（检索增强生成）系统，专为石化/能源行业设计，支持文档精准问答+差分隐私保护，无需显卡、普通CPU即可运行。


## 一、项目简介
针对石化行业**传统文档查询效率低、敏感数据隐私泄露风险高**的痛点，本项目基于大模型RAG技术与差分隐私算法，实现：
- 本地全流程运行：所有数据、模型均在本地处理，无外部网络依赖；
- 精准文档问答：覆盖石化工艺参数、操作规范等场景，核心信息准确率≥90%；
- 隐私安全防护：嵌入层添加可控高斯噪声，避免文档向量数据泄露；
- 轻量低门槛：适配普通CPU环境，无需高端显卡，部署成本低。


## 二、项目结构
LLMStudy/
├── rag_with_dp_BGEembedding.py # 项目主程序（核心逻辑：RAG 链路 + DP 嵌入 + 小模型适配）
├── README.md # 项目说明文档
├── docs/ # 石化文档目录
│ └── petrochemical_operation_manual.txt # 石化行业知识库文档
├── chroma_db/ # Chroma 向量库存储目录（自动生成）
└── models/
└── Qwen2-0.5B-Instruct/ # 本地 Qwen2 小模型目录（需自行下载）



## 三、核心功能特点
1. **本地全隔离运行**：
   - 模型、文档、向量库均在本地存储/计算，无数据上传风险，适配企业敏感数据场景；
2. **差分隐私保护**：
   - 语义嵌入层添加高斯噪声（ε=2.0），平衡“隐私安全”与“检索精度”，检索损失≤5%；
3. **小模型适配优化**：
   - 针对Qwen2-0.5B小模型优化参数（分块大小、生成约束），避免语义断裂、胡编乱造；
4. **轻量低门槛**：
   - 无需显卡，普通CPU（≥4核）即可运行；模型内存占用≤250MB，部署成本极低。


## 四、技术栈
| 技术模块         | 具体工具/框架                     | 作用说明                     |
|------------------|-----------------------------------|------------------------------|
| 编程语言         | Python 3.10+                      | 项目核心开发语言             |
| RAG链路构建      | LangChain                         | 快速搭建文档检索→模型生成链路 |
| 语义嵌入         | BGE-small-zh-v1.5                 | 中文语义精准，轻量适配CPU    |
| 本地大模型       | Qwen2-0.5B-Instruct               | 本地生成回答，无外部API依赖  |
| 向量库           | Chroma DB                         | 轻量本地向量库，存储文档向量 |
| 隐私保护         | 差分隐私（DP）高斯噪声算法        | 保护文档向量数据隐私         |


## 五、详细部署步骤
### 1. 环境准备
#### （1）创建Python环境
推荐使用`conda`创建隔离环境：
```bash
# 创建环境
conda create -n llm-rag python=3.10
# 激活环境
conda activate llm-rag
```

#### （2）安装依赖包
```bash
# 核心依赖（LangChain+向量库）
pip install langchain langchain-community langchain-huggingface
# 嵌入模型依赖
pip install sentence-transformers
# 向量库依赖
pip install chromadb
# 大模型依赖
pip install torch transformers
```
### 2. 下载本地模型
本项目使用Qwen2-0.5B-Instruct小模型（本地运行无 API 费用）：
从阿里云开源模型平台下载模型文件；
将模型解压到项目的models/目录下，路径示例：./models/Qwen2-0.5B-Instruct/。
### 3. 配置项目路径
修改rag_with_dp_BGEembedding.py中的 2 个核心路径：
```python
# 1. 本地Qwen2模型路径（替换为你的实际路径）
LOCAL_QWEN2_PATH = "/home/wangsen/programe/LLMStudy/models/Qwen2-0.5B-Instruct"
# 2. 石化文档路径（默认在docs目录下）
DOC_PATH = "./docs/petrochemical_operation_manual.txt"
```
### 4. 运行项目
在项目根目录执行:
```bash
python rag_with_dp_BGEembedding.py
```
## 六、功能演示
运行后会自动加载文档、构建向量库，并执行测试问答，示例输出:
    ```
    测试问题："石化行业的主要工艺参数有哪些？"
    回答："根据文档，石化行业的主要工艺参数包括：
    1. 压力、温度、流量、流量比等控制参数；
    2. 产品质量参数（如密度、酸度、碱度等）；
    3. 设备性能参数（如效率、噪音、振动等）。"
    ```
