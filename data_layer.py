# ==============================================
# 企业级石化RAG系统 - 数据层
# 核心功能：向量库抽象、Redis缓存、文档处理
# ==============================================

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from rag_dp_llm.config import config
from rag_dp_llm.embeddings import DP_SemanticEmbeddings


class VectorStoreInterface:
    """向量库抽象接口"""
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档"""
        raise NotImplementedError
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """相似度搜索"""
        raise NotImplementedError
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """带分数的相似度搜索"""
        raise NotImplementedError
    
    def delete(self, ids: List[str]) -> None:
        """删除文档"""
        raise NotImplementedError
    
    def clear(self) -> None:
        """清空向量库"""
        raise NotImplementedError


class ChromaVectorStore(VectorStoreInterface):
    """Chroma向量库实现"""
    
    def __init__(self, embeddings: DP_SemanticEmbeddings):
        self.embeddings = embeddings
        self.db_path = os.path.join(config.BASE_DIR, "chroma_db", "dp_chroma_db")
        self._init_db()
    
    def _init_db(self):
        """初始化向量库"""
        if os.path.exists(self.db_path):
            import shutil
            shutil.rmtree(self.db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.vector_store = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings
        )
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        return self.vector_store.add_documents(documents)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def delete(self, ids: List[str]) -> None:
        self.vector_store.delete(ids)
    
    def clear(self) -> None:
        self._init_db()


class OpenSearchVectorStore(VectorStoreInterface):
    """OpenSearch向量库实现"""
    
    def __init__(self, embeddings: DP_SemanticEmbeddings):
        self.embeddings = embeddings
        self.host = config.OPENSEARCH_HOST
        self.port = config.OPENSEARCH_PORT
        self.username = config.OPENSEARCH_USER
        self.password = config.OPENSEARCH_PASSWORD
        self.index_name = config.OPENSEARCH_INDEX
        self._init_client()
    
    def _init_client(self):
        """初始化OpenSearch客户端"""
        try:
            from opensearchpy import OpenSearch
            self.client = OpenSearch(
                hosts=[{'host': self.host, 'port': self.port}],
                http_compress=True,
                http_auth=(self.username, self.password),
                use_ssl=False,
                verify_certs=False,
                ssl_assert_hostname=False,
                ssl_show_warn=False
            )
            self._create_index()
            print("✅ OpenSearch客户端初始化完成")
        except Exception as e:
            print(f"❌ OpenSearch初始化失败: {e}")
            raise
    
    def _create_index(self):
        """创建索引"""
        index_body = {
            'settings': {
                'index': {
                    'number_of_shards': 1,
                    'number_of_replicas': 0
                }
            },
            'mappings': {
                'properties': {
                    'content': {
                        'type': 'text'
                    },
                    'metadata': {
                        'type': 'object'
                    },
                    'embedding': {
                        'type': 'knn_vector',
                        'dimension': config.EMBED_DIM,
                        'method': {
                            'name': 'hnsw',
                            'space_type': 'l2',
                            'engine': 'nmslib'
                        }
                    }
                }
            }
        }
        
        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(index=self.index_name, body=index_body)
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        ids = []
        for i, doc in enumerate(documents):
            embedding = self.embeddings.embed_documents([doc.page_content])[0]
            doc_id = f"doc_{i}_{hash(doc.page_content) % 1000000}"
            
            doc_body = {
                'content': doc.page_content,
                'metadata': doc.metadata,
                'embedding': embedding
            }
            
            self.client.index(
                index=self.index_name,
                body=doc_body,
                id=doc_id
            )
            ids.append(doc_id)
        return ids
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        results = self.similarity_search_with_score(query, k=k)
        return [doc for doc, _ in results]
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        embedding = self.embeddings.embed_query(query)
        
        search_body = {
            'size': k,
            'query': {
                'knn': {
                    'embedding': {
                        'vector': embedding,
                        'k': k
                    }
                }
            }
        }
        
        response = self.client.search(index=self.index_name, body=search_body)
        results = []
        
        for hit in response['hits']['hits']:
            content = hit['_source']['content']
            metadata = hit['_source'].get('metadata', {})
            score = 1.0 / (1.0 + hit['_score'])  # 转换为相似度分数
            doc = Document(page_content=content, metadata=metadata)
            results.append((doc, score))
        
        return results
    
    def delete(self, ids: List[str]) -> None:
        for doc_id in ids:
            try:
                self.client.delete(index=self.index_name, id=doc_id)
            except Exception:
                pass
    
    def clear(self) -> None:
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
        self._create_index()


class RedisCache:
    """Redis缓存实现"""
    
    def __init__(self):
        try:
            import redis
            self.redis_client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                password=config.REDIS_PASSWORD,
                db=config.REDIS_DB,
                decode_responses=True
            )
            self.redis_client.ping()
            print("✅ Redis缓存初始化完成")
        except Exception as e:
            print(f"❌ Redis初始化失败: {e}")
            self.redis_client = None
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if not self.redis_client:
            return None
        
        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            print(f"❌ Redis获取失败: {e}")
        return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """设置缓存"""
        if not self.redis_client:
            return False
        
        try:
            ttl = ttl or config.REDIS_TTL
            self.redis_client.setex(key, ttl, json.dumps(value, ensure_ascii=False))
            return True
        except Exception as e:
            print(f"❌ Redis设置失败: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """删除缓存"""
        if not self.redis_client:
            return False
        
        try:
            self.redis_client.delete(key)
            return True
        except Exception as e:
            print(f"❌ Redis删除失败: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> bool:
        """清除匹配模式的缓存"""
        if not self.redis_client:
            return False
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
            return True
        except Exception as e:
            print(f"❌ Redis清除失败: {e}")
            return False


class DocumentProcessor:
    """文档处理器"""
    
    @staticmethod
    def load_and_split_docs(doc_path: str) -> List[Document]:
        """加载并分割文档"""
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


class VectorStoreFactory:
    """向量库工厂"""
    
    @staticmethod
    def create_vector_store(embeddings: DP_SemanticEmbeddings) -> VectorStoreInterface:
        """创建向量库实例"""
        if config.USE_DOMESTIC_STACK:
            return OpenSearchVectorStore(embeddings)
        else:
            return ChromaVectorStore(embeddings)
