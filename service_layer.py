# ==============================================
# 企业级石化RAG系统 - 服务层
# 核心功能：RAG检索服务、权限服务、故障容错
# ==============================================

import time
import threading
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

from rag_dp_llm.config import config
from rag_dp_llm.data_layer import VectorStoreFactory, RedisCache, DocumentProcessor
from rag_dp_llm.embeddings import DP_SemanticEmbeddings
from rag_dp_llm.llm_interface import LLMFactory
from rag_dp_llm.security import RBACManager, AuditLogger


class RAGService:
    """RAG检索服务"""
    
    def __init__(self):
        self.embeddings = DP_SemanticEmbeddings()
        self.vector_store = VectorStoreFactory.create_vector_store(self.embeddings)
        self.llm = LLMFactory.create_llm()
        self.redis_cache = RedisCache() if config.USE_REDIS_CACHE else None
        self.rbac_manager = RBACManager()
        self.audit_logger = AuditLogger()
        self._init_documents()
    
    def _init_documents(self):
        """初始化文档"""
        docs = DocumentProcessor.load_and_split_docs(config.DOC_PATH)
        self.vector_store.add_documents(docs)
        print("✅ 文档初始化完成")
    
    def _get_cache_key(self, query: str, user_role: str) -> str:
        """生成缓存键"""
        return f"rag:cache:{user_role}:{hash(query) % 1000000}"
    
    def _retrieve_documents(self, query: str, user_role: str,车间: str = None) -> List[Document]:
        """检索文档"""
        # 尝试从缓存获取
        if self.redis_cache:
            cache_key = self._get_cache_key(query, user_role)
            cached_result = self.redis_cache.get(cache_key)
            if cached_result:
                print("✅ 从缓存获取检索结果")
                return [Document(page_content=doc['content'], metadata=doc['metadata']) 
                        for doc in cached_result]
        
        # 执行向量检索
        try:
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query, k=config.TOP_K
            )
            
            # 过滤文档
            filtered_docs = [doc for doc, score in docs_with_scores 
                           if score < (1 - config.SIMILARITY_THRESHOLD)]
            
            if not filtered_docs:
                filtered_docs = [docs_with_scores[0][0]]
            
            # 应用权限过滤
            filtered_docs = self.rbac_manager.filter_documents(
                filtered_docs, user_role, 车间
            )
            
            # 限制上下文大小
            final_docs = filtered_docs[:config.MAX_CONTEXT_SIZE]
            
            # 缓存结果
            if self.redis_cache:
                cache_data = [
                    {'content': doc.page_content, 'metadata': doc.metadata}
                    for doc in final_docs
                ]
                self.redis_cache.set(cache_key, cache_data)
            
            return final_docs
            
        except Exception as e:
            print(f"❌ 向量检索失败，降级到关键词检索: {e}")
            # 降级到关键词检索
            return self._keyword_search(query, user_role, 车间)
    
    def _keyword_search(self, query: str, user_role: str, 车间: str = None) -> List[Document]:
        """关键词检索（降级方案）"""
        # 简单的关键词匹配实现
        docs = DocumentProcessor.load_and_split_docs(config.DOC_PATH)
        
        # 过滤包含关键词的文档
        keywords = query.split()
        filtered_docs = []
        
        for doc in docs:
            content = doc.page_content.lower()
            if any(keyword.lower() in content for keyword in keywords):
                filtered_docs.append(doc)
                if len(filtered_docs) >= config.MAX_CONTEXT_SIZE:
                    break
        
        # 应用权限过滤
        filtered_docs = self.rbac_manager.filter_documents(
            filtered_docs, user_role, 车间
        )
        
        return filtered_docs[:config.MAX_CONTEXT_SIZE]
    
    def generate_answer(self, query: str, user_info: Dict[str, Any]) -> Dict[str, Any]:
        """生成回答"""
        start_time = time.time()
        user_id = user_info.get('user_id', 'anonymous')
        user_role = user_info.get('role', 'operator')
        车间 = user_info.get('车间', None)
        
        try:
            # 权限校验
            if not self.rbac_manager.check_permission(user_role, 'query_rag'):
                return {
                    'answer': '权限不足，无法访问RAG服务',
                    'status': 'error',
                    'error': '权限不足'
                }
            
            # 检索文档
            docs = self._retrieve_documents(query, user_role, 车间)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # 生成回答
            input_dict = {
                'context': context,
                'input': query
            }
            
            answer = self.llm.generate_answer(input_dict)
            
            # 记录审计日志
            self.audit_logger.log_access(
                user_id=user_id,
                action='rag_query',
                resource='petrochemical_knowledge',
                details={
                    'query': query,
                    'context_size': len(docs),
                    'answer_length': len(answer),
                    'response_time': time.time() - start_time
                }
            )
            
            return {
                'answer': answer,
                'status': 'success',
                'context_size': len(docs),
                'response_time': round(time.time() - start_time, 3)
            }
            
        except Exception as e:
            print(f"❌ 生成回答失败: {e}")
            
            # 记录错误日志
            self.audit_logger.log_error(
                user_id=user_id,
                action='rag_query',
                error=str(e)
            )
            
            return {
                'answer': '服务暂时不可用，请稍后重试',
                'status': 'error',
                'error': str(e)
            }


class PermissionService:
    """权限服务"""
    
    def __init__(self):
        self.rbac_manager = RBACManager()
    
    def check_permission(self, user_role: str, permission: str) -> bool:
        """检查权限"""
        return self.rbac_manager.check_permission(user_role, permission)
    
    def get_user_permissions(self, user_role: str) -> List[str]:
        """获取用户权限"""
        return self.rbac_manager.get_role_permissions(user_role)


class FaultToleranceService:
    """故障容错服务"""
    
    def __init__(self):
        self.service_health = {
            'vector_store': True,
            'llm': True,
            'redis': config.USE_REDIS_CACHE
        }
        self.health_check_interval = 30  # 健康检查间隔（秒）
        self._start_health_check()
    
    def _start_health_check(self):
        """启动健康检查"""
        def health_check():
            while True:
                self._check_health()
                time.sleep(self.health_check_interval)
        
        thread = threading.Thread(target=health_check, daemon=True)
        thread.start()
        print("✅ 健康检查服务启动")
    
    def _check_health(self):
        """执行健康检查"""
        # 检查向量存储服务
        try:
            embeddings = DP_SemanticEmbeddings()
            vector_store = VectorStoreFactory.create_vector_store(embeddings)
            vector_store.similarity_search("测试", k=1)
            self.service_health['vector_store'] = True
        except Exception:
            self.service_health['vector_store'] = False
        
        # 检查大模型服务
        try:
            llm = LLMFactory.create_llm()
            llm.generate_answer({'context': '测试', 'input': '测试'})
            self.service_health['llm'] = True
        except Exception:
            self.service_health['llm'] = False
        
        # 检查Redis服务
        if config.USE_REDIS_CACHE:
            try:
                redis = RedisCache()
                redis.set('health_check', 'ok', ttl=10)
                self.service_health['redis'] = True
            except Exception:
                self.service_health['redis'] = False
    
    def get_health_status(self) -> Dict[str, bool]:
        """获取健康状态"""
        return self.service_health
    
    def is_service_available(self, service: str) -> bool:
        """检查服务是否可用"""
        return self.service_health.get(service, False)


class ServiceManager:
    """服务管理器"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.rag_service = RAGService()
            self.permission_service = PermissionService()
            self.fault_tolerance_service = FaultToleranceService()
            self._initialized = True
            print("✅ 服务管理器初始化完成")
    
    def get_rag_service(self) -> RAGService:
        return self.rag_service
    
    def get_permission_service(self) -> PermissionService:
        return self.permission_service
    
    def get_fault_tolerance_service(self) -> FaultToleranceService:
        return self.fault_tolerance_service
