# ==============================================
# 企业级石化生产运维RAG问答系统（适配中石化国企场景）
# 核心定位：适配国企场景的企业级可落地版本，轻量化改造
# 架构设计：三层架构（数据层、服务层、应用层）
# 技术栈：支持国产化替换（OpenSearch + 国产大模型）
# ==============================================
import os
import shutil
import torch
import numpy as np
from typing import List, Dict, Any
import warnings
import json
import time
import multiprocessing
import redis
from concurrent.futures import ThreadPoolExecutor

# -------------------------- 1. 环境配置 --------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="langchain")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 加速模型下载
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 强制CPU运行
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------- 2. 核心依赖导入 --------------------------
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from langchain_huggingface import HuggingFaceEmbeddings

# 国产技术栈依赖（按需导入）
try:
    from opensearchpy import OpenSearch
    from langchain_community.vectorstores import OpenSearchVectorSearch
except ImportError:
    pass

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# -------------------------- 3. 系统配置（支持开关控制） --------------------------
class SystemConfig:
    # 基础配置
    LOCAL_QWEN2_PATH = "/home/wangsen/programe/LLMStudy/models/Qwen2-0.5B-Instruct"
    DOC_PATH = "./docs/petrochemical_operation_manual.txt"
    DP_EPSILON = 2.0
    DP_DELTA = 1e-5
    EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
    SIMILARITY_THRESHOLD = 0.2
    
    # 运行模式开关
    RUN_MODE = "development"  # development: 单机开发版, server: 服务器版
    
    # 国产化替代开关
    USE_DOMESTIC_STACK = False  # True: 使用国产技术栈, False: 使用原技术栈
    
    # Redis缓存开关（仅服务器版生效）
    USE_REDIS_CACHE = False
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_DB = 0
    
    # 多进程服务开关（仅服务器版生效）
    USE_MULTIPROCESS = False
    PROCESS_COUNT = min(4, multiprocessing.cpu_count())
    
    # OpenSearch配置（国产替代时使用）
    OPENSEARCH_HOST = "localhost"
    OPENSEARCH_PORT = 9200
    OPENSEARCH_USER = "admin"
    OPENSEARCH_PASS = "admin"
    OPENSEARCH_INDEX = "petrochemical_knowledge"
    
    # 国产大模型配置
    DOMESTIC_LLM_TYPE = "glm4"  # glm4: 智谱清言, qwen: 通义千问
    GLM4_API_KEY = "your_glm4_api_key"
    QWEN_API_KEY = "your_qwen_api_key"

# -------------------------- 4. 差分隐私嵌入层 --------------------------
class DP_SemanticEmbeddings:
    def __init__(self, embed_model_name: str, epsilon: float = SystemConfig.DP_EPSILON, delta: float = SystemConfig.DP_DELTA):
        self.base_embeddings = HuggingFaceEmbeddings(
            model_name=embed_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.epsilon = epsilon
        self.delta = delta
        self.embed_dim = 768
        self.sigma = np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        np.random.seed(42)

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

# -------------------------- 5. Redis缓存层 --------------------------
class RedisCache:
    def __init__(self):
        if not REDIS_AVAILABLE:
            self.client = None
            return
        
        try:
            self.client = redis.Redis(
                host=SystemConfig.REDIS_HOST,
                port=SystemConfig.REDIS_PORT,
                db=SystemConfig.REDIS_DB,
                decode_responses=True
            )
            self.client.ping()
            print("✅ Redis缓存连接成功")
        except Exception as e:
            print(f"⚠️ Redis连接失败：{e}，将使用本地缓存")
            self.client = None
    
    def get(self, key: str) -> Any:
        if not self.client:
            return None
        
        try:
            value = self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            print(f"⚠️ Redis读取失败：{e}")
            return None
    
    def set(self, key: str, value: Any, expire: int = 3600) -> bool:
        if not self.client:
            return False
        
        try:
            self.client.setex(key, expire, json.dumps(value))
            return True
        except Exception as e:
            print(f"⚠️ Redis写入失败：{e}")
            return False
    
    def delete(self, key: str) -> bool:
        if not self.client:
            return False
        
        try:
            self.client.delete(key)
            return True
        except Exception as e:
            print(f"⚠️ Redis删除失败：{e}")
            return False

# -------------------------- 6. 文档加载与分割 --------------------------
def load_and_split_docs(doc_path: str) -> List[Any]:
    os.makedirs(os.path.dirname(doc_path), exist_ok=True)
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

    loader = TextLoader(doc_path, encoding="utf-8")
    raw_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50,
        separators=["## ", "# ", "\n\n", "\n", "。", "，"]
    )
    split_docs = text_splitter.split_documents(raw_docs)
    print(f"✅ 文档加载完成：共分割为 {len(split_docs)} 个文本块")
    return split_docs

# -------------------------- 7. 向量库构建（支持OpenSearch） --------------------------
def build_vector_db(split_docs: List[Any]) -> Any:
    embeddings = DP_SemanticEmbeddings(SystemConfig.EMBED_MODEL_NAME)
    
    if SystemConfig.USE_DOMESTIC_STACK:
        # 使用OpenSearch
        try:
            print("🔧 开始构建OpenSearch向量库...")
            
            # 连接OpenSearch
            client = OpenSearch(
                hosts=[{'host': SystemConfig.OPENSEARCH_HOST, 'port': SystemConfig.OPENSEARCH_PORT}],
                http_auth=(SystemConfig.OPENSEARCH_USER, SystemConfig.OPENSEARCH_PASS),
                use_ssl=False,
                verify_certs=False,
                connection_class=None
            )
            
            # 检查索引是否存在
            if not client.indices.exists(index=SystemConfig.OPENSEARCH_INDEX):
                # 创建索引
                index_body = {
                    "settings": {
                        "index": {
                            "knn": True,
                            "knn.space_type": "cosinesimil"
                        }
                    },
                    "mappings": {
                        "properties": {
                            "text": {
                                "type": "text"
                            },
                            "vector": {
                                "type": "knn_vector",
                                "dimension": 768,
                                "method": {
                                    "name": "hnsw",
                                    "space_type": "cosinesimil",
                                    "engine": "nmslib"
                                }
                            }
                        }
                    }
                }
                client.indices.create(index=SystemConfig.OPENSEARCH_INDEX, body=index_body)
                print(f"✅ 创建OpenSearch索引：{SystemConfig.OPENSEARCH_INDEX}")
            
            # 构建OpenSearch向量库
            vector_db = OpenSearchVectorSearch(
                embedding_function=embeddings,
                opensearch_url=f"http://{SystemConfig.OPENSEARCH_HOST}:{SystemConfig.OPENSEARCH_PORT}",
                index_name=SystemConfig.OPENSEARCH_INDEX,
                http_auth=(SystemConfig.OPENSEARCH_USER, SystemConfig.OPENSEARCH_PASS)
            )
            
            # 向量化并索引文档
            for i, doc in enumerate(split_docs):
                vector = embeddings.embed_documents([doc.page_content])[0]
                doc_dict = {
                    "text": doc.page_content,
                    "vector": vector
                }
                client.index(index=SystemConfig.OPENSEARCH_INDEX, body=doc_dict, id=i)
            
            print("✅ OpenSearch向量库构建完成")
            return vector_db
        except Exception as e:
            print(f"❌ OpenSearch构建失败，回退到Chroma：{e}")
            # 回退到Chroma
            SystemConfig.USE_DOMESTIC_STACK = False
    
    # 使用Chroma
    db_path = "./chroma_db/dp_chroma_db"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    vector_db = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=db_path
    )
    print("✅ Chroma向量库构建完成")
    return vector_db

# -------------------------- 8. 大模型加载（支持国产模型） --------------------------
class LLMWrapper:
    def __init__(self):
        if SystemConfig.USE_DOMESTIC_STACK:
            # 使用国产大模型
            self.llm_type = SystemConfig.DOMESTIC_LLM_TYPE
            print(f"🔧 初始化国产大模型：{self.llm_type}")
        else:
            # 使用原模型
            self.llm = Qwen2LLM(SystemConfig.LOCAL_QWEN2_PATH)
    
    def generate_answer(self, input_dict: dict) -> str:
        context = input_dict.get("context", "")
        question = input_dict.get("input", "")
        
        if SystemConfig.USE_DOMESTIC_STACK:
            # 调用国产大模型API
            try:
                if self.llm_type == "glm4":
                    # 智谱清言GLM-4调用
                    # 这里使用模拟实现，实际需要根据API文档进行调整
                    print("🤖 调用智谱清言GLM-4")
                    return self._mock_glm4_call(context, question)
                elif self.llm_type == "qwen":
                    # 通义千问Qwen调用
                    print("🤖 调用通义千问Qwen")
                    return self._mock_qwen_call(context, question)
                else:
                    return "未配置有效的国产大模型"
            except Exception as e:
                print(f"❌ 国产大模型调用失败：{e}")
                return "大模型服务暂时不可用"
        else:
            # 使用原模型
            return self.llm.generate_answer(input_dict)
    
    def _mock_glm4_call(self, context: str, question: str) -> str:
        """模拟GLM-4调用"""
        prompt = f"基于以下参考信息回答问题，分点清晰、数值准确，无相关信息时回答'无相关记录'。\n参考信息：{context}\n问题：{question}"
        # 模拟响应
        if "温度" in question:
            return "原油裂化温度约450℃，天然气合成氨核心反应温度450℃"
        elif "用途" in question:
            return "天然气的主要用途：\n1. 民用燃料，用于居民做饭、取暖；\n2. 工业原料，用于生产合成氨、甲醇等化工产品；\n3. 发电燃料，用于燃气轮机发电，效率约55%"
        elif "压力" in question:
            return "原油裂化反应压力5.0MPa"
        else:
            return "无相关记录"
    
    def _mock_qwen_call(self, context: str, question: str) -> str:
        """模拟Qwen调用"""
        return self._mock_glm4_call(context, question)

class Qwen2LLM:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="cpu",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
            pad_token_id=self.tokenizer.pad_token_id
        )
        self.gen_config = GenerationConfig(
            max_new_tokens=300,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.3,
            stop=["<<<|im_end|>"],
        )
        print("✅ Qwen2-0.5B模型加载完成（CPU运行）")

    def generate_answer(self, input_dict: dict) -> str:
        context = input_dict.get("context", "")
        question = input_dict.get("input", "")
        prompt = f"""<<<|im_start|>system
严格基于参考信息回答问题，分点清晰、数值准确，无相关信息时回答"无相关信息"。
参考信息：
{context}
<<<|im_end|>
<<<|im_start|>user
{question}
<<<|im_end|>
<<<|im_start|>assistant
"""
        max_input_length = 1024 - self.gen_config.max_new_tokens
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_input_length
        )
        outputs = self.model.generate(**inputs, generation_config=self.gen_config)
        answer = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        return answer if answer and "无相关信息" not in answer else "无相关记录"

# -------------------------- 9. RBAC权限控制 --------------------------
class RBACManager:
    def __init__(self):
        # 角色定义
        self.roles = {
            "admin": {"access_level": "all", "description": "管理员，可访问全量数据"},
            "operator": {"access_level": "workshop", "description": "生产运维人员，仅可访问本车间数据"}
        }
        # 用户-角色映射
        self.user_roles = {
            "admin_user": "admin",
            "operator_user_1": "operator",
            "operator_user_2": "operator"
        }
        # 用户-车间映射
        self.user_workshops = {
            "operator_user_1": ["refinery"],
            "operator_user_2": ["chemical"]
        }
    
    def check_permission(self, username: str, workshop: str = None) -> bool:
        """检查用户权限"""
        if username not in self.user_roles:
            return False
        
        role = self.user_roles[username]
        if role == "admin":
            return True
        elif role == "operator" and workshop:
            return workshop in self.user_workshops.get(username, [])
        return False
    
    def get_user_access_level(self, username: str) -> str:
        """获取用户访问级别"""
        if username not in self.user_roles:
            return "none"
        return self.roles[self.user_roles[username]]["access_level"]

# -------------------------- 10. 日志审计模块 --------------------------
class AuditLogger:
    def __init__(self):
        self.log_file = "./logs/audit.log"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
    
    def log(self, username: str, action: str, details: dict):
        """记录审计日志"""
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "username": username,
            "action": action,
            "details": details
        }
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            print(f"📋 审计日志已记录：{action}")
        except Exception as e:
            print(f"❌ 审计日志写入失败：{e}")

# -------------------------- 11. 故障容错降级机制 --------------------------
class FaultTolerance:
    def __init__(self):
        self.degraded_mode = False
    
    def check_service_health(self, service_type: str) -> bool:
        """检查服务健康状态"""
        if service_type == "vector_store":
            if SystemConfig.USE_DOMESTIC_STACK:
                # 检查OpenSearch
                try:
                    client = OpenSearch(
                        hosts=[{'host': SystemConfig.OPENSEARCH_HOST, 'port': SystemConfig.OPENSEARCH_PORT}],
                        http_auth=(SystemConfig.OPENSEARCH_USER, SystemConfig.OPENSEARCH_PASS),
                        use_ssl=False,
                        verify_certs=False
                    )
                    client.ping()
                    return True
                except:
                    return False
            else:
                # Chroma本地存储，默认健康
                return True
        elif service_type == "llm":
            # 检查大模型服务
            try:
                if SystemConfig.USE_DOMESTIC_STACK:
                    # 模拟检查
                    return True
                else:
                    # 本地模型，默认健康
                    return True
            except:
                return False
        return True
    
    def degrade_to_keyword_search(self, query: str, documents: List[Any]) -> List[Any]:
        """降级到关键词检索"""
        print("⚠️ 向量检索失败，降级到关键词检索")
        keywords = query.split()
        relevant_docs = []
        
        for doc in documents:
            content = doc.page_content.lower()
            if any(keyword.lower() in content for keyword in keywords):
                relevant_docs.append(doc)
        
        return relevant_docs[:2]  # 最多返回2个文档

# -------------------------- 12. RAG链构建 --------------------------
def build_rag_chain(vector_db: Any, llm: LLMWrapper, redis_cache: RedisCache = None) -> Any:
    fault_tolerance = FaultTolerance()
    
    def retrieve_with_threshold(query: str, username: str = "anonymous", workshop: str = None) -> List[Any]:
        # 生成缓存键
        cache_key = f"rag:query:{hash(query)}:{username}:{workshop}"
        
        # 尝试从缓存获取
        if SystemConfig.USE_REDIS_CACHE and redis_cache:
            cached_result = redis_cache.get(cache_key)
            if cached_result:
                print("✅ 从Redis缓存获取检索结果")
                return cached_result
        
        # 检查向量库健康状态
        if not fault_tolerance.check_service_health("vector_store"):
            # 降级到关键词检索
            # 这里简化处理，实际应从存储中加载文档
            dummy_docs = load_and_split_docs(SystemConfig.DOC_PATH)
            relevant_docs = fault_tolerance.degrade_to_keyword_search(query, dummy_docs)
        else:
            # 正常向量检索
            try:
                if hasattr(vector_db, "similarity_search_with_score"):
                    docs_with_scores = vector_db.similarity_search_with_score(query, k=5)
                    filtered_docs = [doc for doc, score in docs_with_scores if score < (1 - SystemConfig.SIMILARITY_THRESHOLD)]
                    if not filtered_docs:
                        filtered_docs = [docs_with_scores[0][0]]
                    relevant_docs = filtered_docs[:2]
                else:
                    # OpenSearch兼容处理
                    relevant_docs = vector_db.similarity_search(query, k=2)
            except Exception as e:
                print(f"❌ 向量检索失败：{e}")
                # 降级到关键词检索
                dummy_docs = load_and_split_docs(SystemConfig.DOC_PATH)
                relevant_docs = fault_tolerance.degrade_to_keyword_search(query, dummy_docs)
        
        # 缓存结果
        if SystemConfig.USE_REDIS_CACHE and redis_cache:
            redis_cache.set(cache_key, relevant_docs, expire=3600)
        
        return relevant_docs
    
    def generate_with_fallback(input_dict: dict) -> str:
        # 检查大模型健康状态
        if not fault_tolerance.check_service_health("llm"):
            return "大模型服务暂时不可用，请稍后重试"
        
        # 正常生成
        try:
            return llm.generate_answer(input_dict)
        except Exception as e:
            print(f"❌ 大模型生成失败：{e}")
            return "生成回答时出错，请稍后重试"
    
    rag_chain = (
        {
            "context": RunnableLambda(lambda x: retrieve_with_threshold(x["query"], x.get("username"), x.get("workshop"))) | (lambda docs: "\n\n".join([d.page_content for d in docs])),
            "input": RunnableLambda(lambda x: x["query"])
        }
        | RunnableLambda(generate_with_fallback)
        | StrOutputParser()
        | (lambda x: x.strip())
    )
    
    print("✅ RAG链构建完成")
    return rag_chain

# -------------------------- 13. 多进程服务 --------------------------
class RAGService:
    def __init__(self, vector_db: Any, llm: LLMWrapper, redis_cache: RedisCache = None):
        self.vector_db = vector_db
        self.llm = llm
        self.redis_cache = redis_cache
        self.rag_chain = build_rag_chain(vector_db, llm, redis_cache)
        self.rbac = RBACManager()
        self.audit_logger = AuditLogger()
    
    def process_query(self, query: str, username: str = "anonymous", workshop: str = None) -> str:
        """处理单个查询"""
        # 权限检查
        if not self.rbac.check_permission(username, workshop):
            self.audit_logger.log(username, "permission_denied", {"query": query, "workshop": workshop})
            return "权限不足，无法访问该资源"
        
        # 处理查询
        try:
            start_time = time.time()
            result = self.rag_chain.invoke({"query": query, "username": username, "workshop": workshop})
            end_time = time.time()
            
            # 记录审计日志
            self.audit_logger.log(username, "rag_query", {
                "query": query,
                "workshop": workshop,
                "response": result,
                "time_taken": f"{end_time - start_time:.2f}s"
            })
            
            return result
        except Exception as e:
            self.audit_logger.log(username, "query_error", {"query": query, "error": str(e)})
            return f"处理查询时出错：{str(e)}"

# -------------------------- 14. 多进程服务实现 --------------------------
def worker_process(vector_db, llm, redis_cache, task_queue, result_queue):
    """工作进程函数"""
    service = RAGService(vector_db, llm, redis_cache)
    
    while True:
        task = task_queue.get()
        if task is None:
            break
        
        query_id, query, username, workshop = task
        try:
            result = service.process_query(query, username, workshop)
            result_queue.put((query_id, result))
        except Exception as e:
            result_queue.put((query_id, f"处理失败：{str(e)}"))

# -------------------------- 15. 主运行函数 --------------------------
def main():
    try:
        # 加载文档
        split_docs = load_and_split_docs(SystemConfig.DOC_PATH)
        
        # 构建向量库
        vector_db = build_vector_db(split_docs)
        
        # 初始化大模型
        llm = LLMWrapper()
        
        # 初始化Redis缓存
        redis_cache = RedisCache() if SystemConfig.USE_REDIS_CACHE else None
        
        if SystemConfig.RUN_MODE == "development":
            # 单机开发版
            print("🚀 启动单机开发版RAG服务...")
            service = RAGService(vector_db, llm, redis_cache)
            
            # 测试问答
            print("\n========== 开始测试RAG问答 ==========")
            test_questions = [
                "天然气有哪些用途？",
                "金陵石化350万吨炼化装置的核心工艺是什么？",
                "合成氨的反应温度是多少？",
                "原油裂化的反应压力是多少？"
            ]
            
            for idx, question in enumerate(test_questions, 1):
                print(f"\n📝 问题{idx}：{question}")
                answer = service.process_query(question, "admin_user")
                print(f"🤖 回答：{answer}")
                
        else:
            # 服务器版（多进程）
            print("🚀 启动服务器版RAG服务...")
            
            if SystemConfig.USE_MULTIPROCESS:
                # 多进程模式
                task_queue = multiprocessing.Queue()
                result_queue = multiprocessing.Queue()
                
                # 启动工作进程
                processes = []
                for i in range(SystemConfig.PROCESS_COUNT):
                    p = multiprocessing.Process(
                        target=worker_process,
                        args=(vector_db, llm, redis_cache, task_queue, result_queue)
                    )
                    p.start()
                    processes.append(p)
                
                print(f"✅ 启动 {SystemConfig.PROCESS_COUNT} 个工作进程")
                
                # 测试多进程
                test_queries = [
                    (1, "天然气有哪些用途？", "admin_user", None),
                    (2, "合成氨的反应温度是多少？", "operator_user_1", "refinery"),
                    (3, "原油裂化的反应压力是多少？", "operator_user_2", "chemical")
                ]
                
                for task in test_queries:
                    task_queue.put(task)
                
                # 获取结果
                for _ in test_queries:
                    query_id, result = result_queue.get()
                    print(f"\n📝 查询ID：{query_id}")
                    print(f"🤖 结果：{result}")
                
                # 停止工作进程
                for _ in processes:
                    task_queue.put(None)
                
                for p in processes:
                    p.join()
                    
            else:
                # 单进程服务器版
                service = RAGService(vector_db, llm, redis_cache)
                print("✅ 启动单进程服务器版")
                
                # 测试
                test_queries = [
                    ("天然气有哪些用途？", "admin_user", None),
                    ("合成氨的反应温度是多少？", "operator_user_1", "refinery")
                ]
                
                for query, username, workshop in test_queries:
                    print(f"\n📝 查询：{query}")
                    result = service.process_query(query, username, workshop)
                    print(f"🤖 结果：{result}")
        
    except Exception as e:
        print(f"\n❌ 运行出错：{str(e)[:800]}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()