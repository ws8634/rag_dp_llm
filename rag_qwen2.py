import os
import torch

# 1. 文档加载器（社区包）
from langchain_community.document_loaders import PyPDFLoader, TextLoader
# 2. 文本分割器（独立包）
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 3. 向量数据库（社区包）
from langchain_community.vectorstores import Chroma
# 4. 嵌入模型（社区包）
from langchain_community.embeddings import HuggingFaceEmbeddings
# 5. 检索QA链（社区包，最新路径）
from langchain_community.chains import RetrievalQA
# 6. LLM包装器（社区包）
from langchain_community.llms import HuggingFacePipeline
# 7. Transformers相关
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# ====================== 关键配置：强制使用PyTorch，禁用TensorFlow ======================
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["USE_TF"] = "0"
os.environ["USE_PYTORCH"] = "1"

# ====================== 1. 加载并分割本地PDF文档 ======================
def load_and_split_documents():
    # 替换为你的PDF路径（如果没有PDF，先创建一个简单的txt文档测试）
    pdf_path = "./docs/测试文档.pdf"
    
    # 检查文件是否存在（新手友好：没有PDF就用TXT替代）
    if not os.path.exists(pdf_path):
        # 自动创建测试文档
        os.makedirs("./docs", exist_ok=True)
        with open("./docs/测试文档.txt", "w", encoding="utf-8") as f:
            f.write("石化生产的主要原料包括原油、天然气、煤炭等。\n")
            f.write("原油经过蒸馏、裂化等工艺，可生产出汽油、柴油、乙烯等产品。\n")
        # 改用TXT加载器（避免PDF依赖问题）
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader("./docs/测试文档.txt", encoding="utf-8")
    else:
        loader = PyPDFLoader(pdf_path)
    
    documents = loader.load()
    
    # 分割文本（适配CPU运行）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,    # 减小块大小，降低CPU内存占用
        chunk_overlap=20,
        length_function=len
    )
    splits = text_splitter.split_documents(documents)
    print(f"文档分割完成，共生成 {len(splits)} 个文本块")
    return splits

# ====================== 2. 构建向量数据库（轻量化配置） ======================
def build_vector_db(splits):
    # 加载轻量级向量模型（禁用CUDA，纯CPU）
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # 构建向量数据库（本地存储）
    vector_db = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name="test_docs"
    )
    vector_db.persist()
    print("向量数据库构建完成")
    return vector_db

# ====================== 3. 加载Qwen2模型（纯CPU优化） ======================
def build_qwen2_llm():
    # 选择极小模型，适配CPU运行
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    # 加载Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型（纯CPU，低内存模式）
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="cpu",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32  # 用float32降低内存占用
    )
    
    # 构建生成Pipeline（CPU优化）
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,    # 减小生成长度，加快速度
        temperature=0.1,       # 降低随机性，提升回答准确性
        do_sample=False,       # 关闭采样，纯CPU更快
        pad_token_id=tokenizer.eos_token_id,
        device_map="cpu"
    )
    
    # 包装成LangChain LLM
    llm = HuggingFacePipeline(pipeline=pipe)
    print("Qwen2模型加载完成（纯CPU模式）")
    return llm

# ====================== 4. 构建RAG问答链 ======================
def build_rag_chain(vector_db, llm):
    # 构建检索问答链（适配CPU）
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(k=2),  # 减少检索数量，加快速度
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": """基于以下参考文档回答问题，只使用文档中的信息，不要编造：
{context}

问题：{question}
回答："""
        }
    )
    print("RAG问答链构建完成")
    return qa_chain

# ====================== 主流程 ======================
if __name__ == "__main__":
    # 1. 加载文档
    splits = load_and_split_documents()
    # 2. 构建向量库
    vector_db = build_vector_db(splits)
    # 3. 加载模型
    llm = build_qwen2_llm()
    # 4. 构建问答链
    qa_chain = build_rag_chain(vector_db, llm)
    
    # 测试问答
    query = "石化生产的主要原料有哪些？"
    print(f"\n📝 提问：{query}")
    try:
        result = qa_chain.invoke(query)  # 改用invoke（新版LangChain推荐）
        print(f"🤖 回答：{result['result'].strip()}")
        # 打印参考文档
        print("\n🔍 参考文档：")
        for i, doc in enumerate(result["source_documents"]):
            print(f"{i+1}. {doc.page_content.strip()}")
    except Exception as e:
        print(f"❌ 运行出错：{str(e)[:200]}")