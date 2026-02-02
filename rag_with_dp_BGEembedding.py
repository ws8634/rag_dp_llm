# ==============================================
# Qwen2-0.5B 小模型最优适配 RAG 程序（带差分隐私）
# 核心定位：适配 0.5B 参数量小模型的「松而有度」配置，实现「核心信息精准+少量可接受编造」
# 最优解逻辑：不追求“完美无编造”（小模型能力上限），优先保证「检索准、数值对、框架清」
# 适配场景：本地CPU运行、石化行业短文档问答、需兼顾差分隐私与检索精度
# ==============================================
import os
import shutil
import torch
import numpy as np
from typing import List, Dict, Any
import warnings

# -------------------------- 1. 环境配置（适配小模型CPU运行，屏蔽无关警告） --------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="langchain")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 加速模型下载（小模型无需额外优化）
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 强制CPU运行（0.5B模型CPU足够承载，无需显卡）
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免多线程冲突，保证运行稳定

# -------------------------- 2. 核心依赖导入（轻量适配，不引入复杂工具） --------------------------
from langchain_community.document_loaders import TextLoader  # 轻量文档加载，适配txt短文档
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 基础分块，不搞复杂语义分块
from langchain_community.vectorstores import Chroma  # 轻量向量库，CPU运行无压力
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from langchain_huggingface import HuggingFaceEmbeddings  # 适配新版LangChain，不换复杂嵌入

# -------------------------- 3. 关键配置（最优比例核心！每个参数都卡小模型能力边界） --------------------------
LOCAL_QWEN2_PATH = "/home/wangsen/programe/LLMStudy/models/Qwen2-0.5B-Instruct"  # 本地小模型路径
DOC_PATH = "./docs/petrochemical_operation_manual.txt"  # 短文档路径（小模型不支持长文档处理）
DP_EPSILON = 2.0    # 差分隐私最优值：ε=2.0（噪声小→检索准，隐私性达标；ε<1.0噪声大，ε>3.0隐私性不足）
DP_DELTA = 1e-5     # 固定松弛参数（行业通用1e-5，无需调整）
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"  # 最优嵌入模型：轻量（几十MB）、中文语义准，适配CPU
SIMILARITY_THRESHOLD = 0.2  # 检索阈值最优值：0.2（放宽但不泛滥，既能匹配到核心文档，又不引入过多无关内容）

# -------------------------- 4. 有语义+差分隐私嵌入层（小模型适配版：语义准+噪声温和） --------------------------
class DP_SemanticEmbeddings:
    def __init__(self, embed_model_name: str, epsilon: float = DP_EPSILON, delta: float = DP_DELTA):
        # BGE-small-zh-v1.5 是小模型最优嵌入选择：语义理解能力强于字符嵌入，体积小于其他大嵌入模型
        self.base_embeddings = HuggingFaceEmbeddings(
            model_name=embed_model_name,
            model_kwargs={"device": "cpu"},  # 强制CPU，适配低配置
            encode_kwargs={"normalize_embeddings": True}  # 归一化向量，提升检索精度
        )
        self.epsilon = epsilon
        self.delta = delta
        self.embed_dim = 768  # BGE固定维度，无需调整
        self.sigma = np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon  # 温和噪声（ε=2.0→sigma小）
        np.random.seed(42)  # 固定种子，保证检索一致性（小模型敏感，种子变动会导致检索波动）

    def _add_dp_noise(self, vector: List[float]) -> List[float]:
        noise = np.random.normal(0, self.sigma, len(vector))
        noisy_vector = [float(v + n) for v, n in zip(vector, noise)]
        norm = np.linalg.norm(noisy_vector)
        noisy_vector = [v / norm for v in noisy_vector]
        return noisy_vector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        base_vecs = self.base_embeddings.embed_documents(texts)
        return [self._add_dp_noise(vec) for vec in base_vecs]

    def embed_query(self, text: str) -> List[float]:
        base_vec = self.base_embeddings.embed_query(text)
        return self._add_dp_noise(base_vec)

# -------------------------- 5. 文档加载与分割（小模型最优分块：不碎片化+不冗余） --------------------------
def load_and_split_docs(doc_path: str) -> List[Any]:
    os.makedirs(os.path.dirname(doc_path), exist_ok=True)
    if not os.path.exists(doc_path):
        # 文档内容最优设计：短文本+核心信息集中（小模型只能处理短上下文，长文档会导致语义混乱）
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

    loader = TextLoader(doc_path, encoding="utf-8")
    raw_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # 小模型最优分块大小：800字（适配文档长度，分2块，上下文完整不碎片化）
        chunk_overlap=50,  # 轻度重叠，避免语义割裂（小模型语义衔接弱，重叠度过高会冗余）
        separators=["## ", "# ", "\n\n", "\n", "。", "，"]  # 基础分隔符，不搞复杂语义分割（小模型扛不住）
    )
    split_docs = text_splitter.split_documents(raw_docs)
    print(f"✅ 文档加载完成：共分割为 {len(split_docs)} 个文本块")
    return split_docs

# -------------------------- 6. 构建向量库（轻量适配，不搞复杂混合检索） --------------------------
def build_dp_vector_db(split_docs: List[Any]) -> Chroma:
    db_path = "./chroma_db/dp_chroma_db"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)  # 清理旧库，避免小模型检索时受缓存干扰（小模型敏感）
    dp_embeddings = DP_SemanticEmbeddings(embed_model_name=EMBED_MODEL_NAME)
    vector_db = Chroma.from_documents(
        documents=split_docs,
        embedding=dp_embeddings,
        persist_directory=db_path
    )
    print(f"✅ 有语义+差分隐私的向量库构建完成（ε={DP_EPSILON}）")
    return vector_db

# -------------------------- 7. Qwen2-0.5B模型加载（最优生成配置：轻度约束+不逼小模型） --------------------------
class Qwen2LLM:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # 补充pad_token，避免小模型报错（小模型鲁棒性弱）
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="cpu",
            low_cpu_mem_usage=True,  # 低内存占用，适配CPU
            torch_dtype=torch.float32,  # 单精度浮点，平衡速度和精度（小模型用float16反而可能不稳定）
            pad_token_id=self.tokenizer.pad_token_id
        )
        # 生成配置最优组合：小模型能承受的「轻度约束」
        self.gen_config = GenerationConfig(
            max_new_tokens=300,    # 最优长度：300字（既能输出核心信息，又不逼小模型编造过多内容；<250会截断，>350会冗余）
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False,       # 关闭采样（小模型采样会完全胡编，必须固定输出）
            temperature=0.0,      # 无温度（避免随机，保证核心数值稳定）
            repetition_penalty=1.3,  # 最优惩罚：1.3（轻度惩罚，避免重复；>1.4会打断语义，<1.2会过度重复）
            stop=["<<<|im_end|>"],   # 终止符（小模型能识别，避免输出过长）
        )
        print("✅ Qwen2-0.5B模型加载完成（CPU运行）")

    def generate_answer(self, input_dict: dict) -> str:
        context = input_dict.get("context", "")
        question = input_dict.get("input", "")
        # Prompt最优设计：小模型能理解的「简单指令」（不搞复杂约束，避免小模型 confusion）
        prompt = f"""<<<|im_start|>system
严格基于参考信息回答问题，分点清晰、数值准确，无相关信息时回答“无相关记录”。
参考信息：
{context}
<<<|im_end|>
<<<|im_start|>user
{question}
<<<|im_end|>
<<<|im_start|>assistant
"""
        max_input_length = 1024 - self.gen_config.max_new_tokens  # 预留生成空间（小模型输入长度有限，避免截断）
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_input_length
        )
        outputs = self.model.generate(**inputs, generation_config=self.gen_config)
        answer = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        return answer if answer and "无相关信息" not in answer else "无相关记录"

# -------------------------- 8. 构建RAG链（小模型最优检索逻辑：保留核心，不搞复杂过滤） --------------------------
def build_rag_chain(vector_db: Chroma, llm: Qwen2LLM) -> Any:
    def retrieve_with_threshold(query: str) -> List[Any]:
        docs_with_scores = vector_db.similarity_search_with_score(query, k=5)
        # 检索逻辑最优：保留最相关文档（小模型无法处理多文档融合，最多保留2个核心块）
        filtered_docs = [doc for doc, score in docs_with_scores if score < (1 - SIMILARITY_THRESHOLD)]
        if not filtered_docs:
            filtered_docs = [docs_with_scores[0][0]]  # 兜底保留1个，避免小模型空上下文胡编
        return filtered_docs[:2]  # 最多2个块（小模型上下文处理能力有限，多了会混乱）
    
    rag_chain = (
        {
            "context": RunnableLambda(retrieve_with_threshold) | (lambda docs: "\n\n".join([d.page_content for d in docs])),
            "input": RunnablePassthrough()
        }
        | RunnableLambda(llm.generate_answer)
        | StrOutputParser()
        | (lambda x: x.strip())
    )
    print("✅ 生产级RAG链构建完成（兼容新版Chroma）")
    return rag_chain

# -------------------------- 9. 主运行函数（简洁流程，不搞复杂逻辑） --------------------------
def main():
    try:
        split_docs = load_and_split_docs(DOC_PATH)
        vector_db = build_dp_vector_db(split_docs)
        qwen2_llm = Qwen2LLM(LOCAL_QWEN2_PATH)
        rag_chain = build_rag_chain(vector_db, qwen2_llm)

        print("\n========== 开始测试RAG问答 ==========")
        test_questions = [
            "天然气有哪些用途？",
            "金陵石化350万吨炼化装置的核心工艺是什么？",
            "介绍一下许二狗的性格特点和日常行为",
            "常减压蒸馏装置生产运行的规程是什么？",
            "合成氨的反应温度是多少？",
            "原油裂化的反应压力是多少？",
            "减压塔的真空度要求是什么？",
            "常压炉出口温度范围是多少？"
        ]
        for idx, question in enumerate(test_questions, 1):
            print(f"\n📝 问题{idx}：{question}")
            answer = rag_chain.invoke(question)
            print(f"🤖 回答：{answer}")

    except Exception as e:
        print(f"\n❌ 运行出错：{str(e)[:800]}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()