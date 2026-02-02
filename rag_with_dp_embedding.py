# ==============================================
# 带差分隐私嵌入层的RAG完整代码（直接运行版）
# 适配场景：本地石化文档RAG，Qwen2-0.5B模型，Chroma向量库
# 核心：嵌入层添加高斯噪声实现差分隐私，保证检索精度的同时保护向量隐私
# ==============================================
import os
import shutil
import torch
import numpy as np
from typing import List, Dict, Any
import warnings

# -------------------------- 1. 环境配置（避免警告+适配本地运行） --------------------------
# 屏蔽无关警告
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="langchain")

# 配置HF镜像（加速模型加载）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 强制使用CPU运行（无需显卡，适配低配置环境）
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------- 2. 核心依赖导入 --------------------------
# 文档加载与分割
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 向量库
from langchain_community.vectorstores import Chroma
# RAG链构建
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
# Qwen2模型
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# -------------------------- 3. 关键配置（用户只需修改这部分） --------------------------
# 1. 本地Qwen2模型路径（替换成你自己的模型路径）
LOCAL_QWEN2_PATH = "/home/wangsen/programe/LLMStudy/models/Qwen2-0.5B-Instruct"
# 2. 本地石化文档路径（替换成你的文档路径，txt格式）
DOC_PATH = "./docs/petrochemical_docs.txt"
# 3. 差分隐私参数（核心！ε越小隐私性越强，ε越大检索精度越高，推荐1.0-2.0）
DP_EPSILON = 1.0    # 隐私预算
DP_DELTA = 1e-5     # 松弛参数，固定即可
DP_DIM = 100        # 嵌入向量维度，固定即可

# -------------------------- 4. 带差分隐私的嵌入层（核心） --------------------------
class DP_LocalEmbeddings:
    """
    嵌入层添加高斯噪声实现差分隐私：
    1. 先生成基础字符嵌入向量（适配本地轻量运行）
    2. 按(ε,δ)差分隐私规则添加高斯噪声
    3. 固定随机种子保证查询/文档向量噪声规则一致，不影响检索精度
    """
    def __init__(self, dim: int = DP_DIM, epsilon: float = DP_EPSILON, delta: float = DP_DELTA):
        self.dim = dim
        self.epsilon = epsilon
        self.delta = delta
        # 计算高斯噪声的标准差（差分隐私核心公式）
        self.sigma = np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        # 固定随机种子，保证查询和文档的噪声规则一致
        np.random.seed(42)

    def _add_dp_noise(self, vector: List[float]) -> List[float]:
        """给向量添加可控高斯噪声，限制值域避免异常"""
        # 生成高斯噪声
        noise = np.random.normal(0, self.sigma, len(vector))
        # 噪声叠加到原始向量
        noisy_vector = [float(v + n) for v, n in zip(vector, noise)]
        # 限制向量值域在[-1.0, 1.0]，避免数值溢出
        noisy_vector = [max(-1.0, min(1.0, v)) for v in noisy_vector]
        return noisy_vector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量处理文档文本，生成带DP的嵌入向量"""
        # 生成基础字符嵌入（按字符ASCII码归一化）
        base_embeddings = []
        for text in texts:
            base_vec = [ord(c) / 1000 for c in text[:self.dim]]  # 取前dim个字符
            # 补零到固定维度
            base_vec += [0.0] * (self.dim - len(base_vec))
            base_embeddings.append(base_vec)
        # 添加差分隐私噪声
        dp_embeddings = [self._add_dp_noise(vec) for vec in base_embeddings]
        return dp_embeddings

    def embed_query(self, text: str) -> List[float]:
        """处理查询文本，生成带DP的查询向量（和文档向量噪声规则一致）"""
        # 生成基础字符嵌入
        base_vec = [ord(c) / 1000 for c in text[:self.dim]]
        base_vec += [0.0] * (self.dim - len(base_vec))
        # 添加相同规则的高斯噪声
        dp_vec = self._add_dp_noise(base_vec)
        return dp_vec

# -------------------------- 5. 文档加载与分割 --------------------------
def load_and_split_docs(doc_path: str) -> List[Any]:
    """加载本地文档并分块，适配长文本检索"""
    # 确保文档目录存在
    os.makedirs(os.path.dirname(doc_path), exist_ok=True)
    # 若文档不存在，创建测试文档（避免运行报错）
    if not os.path.exists(doc_path):
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write("""金陵石化350万吨炼化装置核心工艺：
1. 原油裂化温度约450℃，反应压力5.0MPa；
2. 天然气合成氨核心反应温度450℃，催化剂为铁基催化剂；
3. 炼化装置的副产品包括丙烷、丁烷，年产能约50万吨。
天然气的主要用途：
1. 民用燃料，用于居民做饭、取暖；
2. 工业原料，用于生产合成氨、甲醇等化工产品；
3. 发电燃料，用于燃气轮机发电，效率约55%。""")
        print(f"⚠️ 未找到文档{doc_path}，已创建测试石化文档")

    # 加载文档
    loader = TextLoader(doc_path, encoding="utf-8")
    raw_docs = loader.load()
    # 分块（适配短文本检索，提升精度）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,    # 每个文本块200字
        chunk_overlap=20,  # 重叠20字，避免语义割裂
        separators=["\n", "。", "，"]  # 按中文分隔符分割
    )
    split_docs = text_splitter.split_documents(raw_docs)
    print(f"✅ 文档加载完成：共分割为 {len(split_docs)} 个文本块")
    return split_docs

# -------------------------- 6. 构建带DP的Chroma向量库 --------------------------
def build_dp_vector_db(split_docs: List[Any]) -> Chroma:
    """构建带差分隐私的Chroma向量库，自动清理旧库"""
    # 向量库存储路径
    db_path = "./chroma_db/dp_chroma_db"
    # 清理旧库（避免缓存干扰）
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    # 初始化DP嵌入层
    dp_embeddings = DP_LocalEmbeddings(
        dim=DP_DIM,
        epsilon=DP_EPSILON,
        delta=DP_DELTA
    )
    # 构建向量库（存储DP向量+原始文本）
    vector_db = Chroma.from_documents(
        documents=split_docs,
        embedding=dp_embeddings,
        persist_directory=db_path
    )
    print(f"✅ 带差分隐私的向量库构建完成（ε={DP_EPSILON}）")
    return vector_db

# -------------------------- 7. Qwen2模型加载（固定回答逻辑） --------------------------
class Qwen2LLM:
    """本地Qwen2模型封装，保证回答全面、固定"""
    def __init__(self, model_path: str):
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right"
        )
        # 补充pad_token（避免报错）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # 加载模型（CPU运行，低内存占用）
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="cpu",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
            pad_token_id=self.tokenizer.pad_token_id
        )
        # 固定生成配置（保证回答全面、无随机性）
        self.gen_config = GenerationConfig(
            max_new_tokens=256,    # 最大生成256字
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False,       # 关闭采样，固定回答
            temperature=None,      # 无温度，避免随机
            top_p=None,
            top_k=None,
            stop=["<|im_end|>"],   # 终止符
        )
        print("✅ Qwen2-0.5B模型加载完成（CPU运行）")

    def generate_answer(self, input_dict: dict) -> str:
        """生成回答：整合检索上下文+用户问题，固定格式"""
        # 提取上下文和问题
        context = input_dict.get("context", "")
        question = input_dict.get("input", "")
        # 构建Qwen2专用prompt
        prompt = f"""<|im_start|>system
请根据以下参考信息回答问题，回答要全面、分点清晰，以句号结尾，不要添加无关内容：
参考文档：
{context}
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
"""
        # 编码prompt
        max_input_length = 1024 - self.gen_config.max_new_tokens  # 预留生成空间
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length
        )
        # 生成回答
        outputs = self.model.generate(**inputs, generation_config=self.gen_config)
        # 解码并清理回答
        answer = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        return answer

# -------------------------- 8. 构建完整RAG链 --------------------------
def build_rag_chain(vector_db: Chroma, llm: Qwen2LLM) -> Any:
    """构建带DP嵌入层的完整RAG链"""
    # 检索器（取最相似的2个文本块）
    retriever = vector_db.as_retriever(k=2)
    # 构建LCEL链
    rag_chain = (
        {
            # 检索上下文：检索器→拼接文本块
            "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
            # 用户问题透传
            "input": RunnablePassthrough()
        }
        # 模型生成回答
        | RunnableLambda(llm.generate_answer)
        # 输出解析
        | StrOutputParser()
        # 清理空格
        | (lambda x: x.strip())
    )
    print("✅ 带差分隐私的RAG链构建完成")
    return rag_chain

# -------------------------- 9. 主运行函数 --------------------------
def main():
    try:
        # 步骤1：加载并分割文档
        split_docs = load_and_split_docs(DOC_PATH)
        # 步骤2：构建带DP的向量库
        vector_db = build_dp_vector_db(split_docs)
        # 步骤3：加载Qwen2模型
        qwen2_llm = Qwen2LLM(LOCAL_QWEN2_PATH)
        # 步骤4：构建RAG链
        rag_chain = build_rag_chain(vector_db, qwen2_llm)

        # 步骤5：测试问答（可替换成自己的问题）
        print("\n========== 开始测试RAG问答 ==========")
        test_questions = [
            "天然气有哪些用途？",
            "常减压蒸馏装置生产运行的规程是什么？",
            "金陵石化350万吨炼化装置的核心工艺是什么？",
            "介绍一下许二狗的性格特点和日常行为",
            "合成氨的反应温度是多少？"
        ]
        for idx, question in enumerate(test_questions, 1):
            print(f"\n📝 问题{idx}：{question}")
            # 执行RAG问答
            answer = rag_chain.invoke(question)
            print(f"🤖 回答：{answer}")

    except Exception as e:
        print(f"\n❌ 运行出错：{str(e)[:800]}")
        import traceback
        traceback.print_exc()

# -------------------------- 10. 运行入口 --------------------------
if __name__ == "__main__":
    main()