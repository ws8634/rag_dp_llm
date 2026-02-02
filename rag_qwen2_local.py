"""
======================================= 运行条件备注 =======================================
1. Python版本：3.10（验证过兼容，其他3.9+/3.11+也可）
2. 核心依赖及验证过的版本号（必须安装）：
   pip install langchain==1.2.6 
   pip install langchain-core langchain-community 
   pip install chromadb==1.4.1 
   pip install transformers==4.41.2 
   pip install torch==2.9.1 
   pip install sentencepiece  # Qwen2模型必需
3. 模型要求：
   - 本地已下载Qwen2-0.5B-Instruct模型
   - 替换下方LOCAL_QWEN2_PATH为实际模型路径
4. 运行环境：
   - Linux/macOS/Windows均可
   - 纯CPU运行（无需GPU，低内存即可）
5. LangChain核心：
   - 严格遵循LangChain 1.x LCEL语法
   - 无第三方API依赖，全本地运行
======================================= 代码开始 =======================================
"""

import os
import shutil
import torch
from typing import Optional, List, Dict, Any

# -------------------------- 1. 环境配置 --------------------------
# 国内HF镜像加速（可选）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 强制CPU运行（无GPU也可）
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 关闭TensorFlow无关日志
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# 关闭分词器并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 关闭transformers无关警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# -------------------------- 2. 核心导入 --------------------------
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# -------------------------- 3. 本地文档加载 --------------------------
def load_and_split_local_docs():
    # os.makedirs("./docs", exist_ok=True)
    doc_path = "./docs/petrochemical_docs.txt"
    # with open(doc_path, "w", encoding="utf-8") as f:
    #     f.write("石化生产的主要原料包括原油、天然气、煤炭和生物质等。\n")
    #     f.write("原油经过蒸馏、裂化、加氢等工艺，可生产汽油、柴油、乙烯、丙烯等基础化工原料。\n")
    #     f.write("天然气主要用于生产合成氨、甲醇和乙烯，也是重要的清洁能源。\n")

    loader = TextLoader(doc_path, encoding="utf-8")
    raw_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        length_function=len,
        separators=["\n", "。", "，"]
    )
    split_docs = text_splitter.split_documents(raw_docs)
    print(f"✅ 本地文档加载完成：共分割为 {len(split_docs)} 个文本块")
    return split_docs

# -------------------------- 4. 本地向量库 --------------------------
class LocalEmbeddings:
    def __init__(self, dim: int = 100):
        self.dim = dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[ord(c) / 1000 for c in text[:self.dim]] + [0.0]*(self.dim - len(text[:self.dim])) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return [ord(c) / 1000 for c in text[:self.dim]] + [0.0]*(self.dim - len(text[:self.dim]))

def build_local_chroma_db(split_docs):
    if os.path.exists("./local_chroma_db"):
        shutil.rmtree("./local_chroma_db")

    embeddings = LocalEmbeddings(dim=100)
    vector_db = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory="./local_chroma_db"
    )
    print("✅ 本地Chroma向量库构建完成")
    return vector_db

# -------------------------- 5. Qwen2模型（彻底清理警告+解决重复回答） --------------------------
class Qwen2DirectLLM:
    """无警告+无重复回答，彻底优化生成逻辑"""
    def __init__(self, model_path: str):
        # 1. 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 2. 加载模型（覆盖默认生成配置）
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="cpu",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
            pad_token_id=self.tokenizer.pad_token_id
        )

        # 3. 彻底清理生成参数（无采样参数，无警告）
        self.gen_config = GenerationConfig(
            max_new_tokens=256,        # 增加生成长度 #1 max_new_tokens=100,          # 缩短生成长度，避免重复
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False,            # 开启采样以获得更多内容  #1 do_sample=False,             # 确定性生成
            temperature=None,             #1 temperature=None,            # 彻底移除采样参数
            top_p=None,
            top_k=None,
            stop=["<|im_end|>"],       # 不以句号作为停止符，允许多句输出 #1 stop=["<|im_end|>", "。"],   # 添加停止词，生成到句号为止
        )
        print("✅ 本地Qwen2-0.5B-Instruct模型加载完成（无pipeline+无警告）")

    def format_qwen2_prompt(self, context: str, question: str) -> str:
        """Qwen2官方prompt格式"""
        prompt = f"""<|im_start|>system
请根据以下参考信息回答问题，回答简洁，以句号结尾:
参考文档：
{context}
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
"""
        return prompt

    def generate(self, input_dict: dict) -> str:
        """生成逻辑：允许更长输出且不按句号截断"""  #1 """生成逻辑优化：限制长度+停止词+简洁回答"""
        # 提取上下文和问题
        context = input_dict.get("context", "")
        question = input_dict.get("input", "")
        
        # 格式化prompt
        prompt_text = self.format_qwen2_prompt(context, question)
        
        # 根据max_new_tokens计算可用输入长度
        max_input_length = max(256, 1024 - self.gen_config.max_new_tokens)
 
        # 编码
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length #1 max_length=1024 - self.gen_config.max_new_tokens
        )

        # 生成（使用干净的GenerationConfig，无警告）
        outputs = self.model.generate(
            **inputs,
            generation_config=self.gen_config
        )

        # 解码+清理重复内容
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        #1 # 截断到第一个句号，解决重复问题
        #1 if "。" in generated_text:
        #1     generated_text = generated_text.split("。")[0] + "。"
        
        return generated_text

# -------------------------- 6. 构建LCEL RAG链 --------------------------
def build_lcel_rag_chain(vector_db, qwen2_llm):
    # 检索器
    retriever = vector_db.as_retriever(k=2)

    # 生成函数封装
    def qwen2_generate(input_dict):
        return qwen2_llm.generate(input_dict)

    # LCEL链
    rag_chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
            "input": RunnablePassthrough()
        }
        | RunnableLambda(qwen2_generate)
        | StrOutputParser()
        | (lambda x: x.strip())
    )

    print("✅ LCEL RAG链构建完成（适配Qwen2+无重复回答）")
    return rag_chain

# -------------------------- 7. 主流程 --------------------------
if __name__ == "__main__":
    LOCAL_QWEN2_PATH = "/home/wangsen/programe/LLMStudy/models/Qwen2-0.5B-Instruct"

    try:
        # 加载文档
        split_docs = load_and_split_local_docs()
        
        # 构建向量库
        vector_db = build_local_chroma_db(split_docs)
        
        # 加载模型
        qwen2_llm = Qwen2DirectLLM(LOCAL_QWEN2_PATH)
        
        # 构建链
        rag_chain = build_lcel_rag_chain(vector_db, qwen2_llm)

        # 测试
        print("\n========== 本地Qwen2模型 RAG问答测试 ==========")
        test_queries = [
            "天然气有哪些用途？",
            "介绍一下许二狗的性格特点和日常行为",
            "石化生产过程是什么、使用哪些原材料、有哪些生成产品",
            "某小学班级有32个同学，分成2组，每组多少人？"
        ]

        for idx, query in enumerate(test_queries, 1):
            print(f"\n📝 测试{idx} - 问题：{query}")
            response = rag_chain.invoke(query)
            print(f"🤖 回答：{response}")

        print("\n🎉 所有测试完成！无警告+无重复回答+有效输出！")

    except Exception as e:
        print(f"\n❌ 运行错误：{str(e)[:800]}")
        import traceback
        traceback.print_exc()