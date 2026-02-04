# ==============================================
# 企业级石化RAG系统配置文件
# 核心功能：管理技术栈选择、开关控制、系统参数
# ==============================================

import os
from typing import Dict, Any


class Config:
    """系统配置类"""
    
    # 基础配置
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DOC_PATH = os.path.join(BASE_DIR, "docs", "petrochemical_operation_manual.txt")
    
    # 技术栈选择开关
    USE_DOMESTIC_STACK = os.environ.get("USE_DOMESTIC_STACK", "false").lower() == "true"  # 国产化技术栈开关
    USE_REDIS_CACHE = os.environ.get("USE_REDIS_CACHE", "false").lower() == "true"  # Redis缓存开关
    USE_MULTIPROCESS = os.environ.get("USE_MULTIPROCESS", "false").lower() == "true"  # 多进程服务开关
    
    # 环境模式
    ENV_MODE = os.environ.get("ENV_MODE", "development")  # development: 单机开发版, production: 服务器版
    
    # 嵌入模型配置
    EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"  # 嵌入模型
    EMBED_DIM = 768  # 嵌入维度
    
    # 差分隐私配置
    DP_EPSILON = 2.0  # 差分隐私ε值
    DP_DELTA = 1e-5  # 差分隐私δ值
    
    # 检索配置
    SIMILARITY_THRESHOLD = 0.2  # 相似度阈值
    TOP_K = 5  # 检索top-k
    MAX_CONTEXT_SIZE = 2  # 最大上下文块数
    
    # 大模型配置
    LOCAL_QWEN2_PATH = "/home/wangsen/programe/LLMStudy/models/Qwen2-0.5B-Instruct"  # 本地Qwen2模型路径
    
    # 国产大模型配置
    GLM4_API_KEY = os.environ.get("GLM4_API_KEY", "")  # 智谱清言API密钥
    GLM4_API_URL = "https://open.bigmodel.cn/api/messages"  # 智谱清言API地址
    
    QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")  # 通义千问API密钥
    QWEN_API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"  # 通义千问API地址
    
    # OpenSearch配置
    OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST", "localhost")
    OPENSEARCH_PORT = int(os.environ.get("OPENSEARCH_PORT", "9200"))
    OPENSEARCH_USER = os.environ.get("OPENSEARCH_USER", "admin")
    OPENSEARCH_PASSWORD = os.environ.get("OPENSEARCH_PASSWORD", "admin")
    OPENSEARCH_INDEX = "petrochemical_knowledge"
    
    # Redis配置
    REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
    REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
    REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
    REDIS_TTL = int(os.environ.get("REDIS_TTL", "3600"))  # 缓存过期时间（秒）
    
    # 多进程配置
    WORKER_COUNT = int(os.environ.get("WORKER_COUNT", "4"))  # 工作进程数
    
    # 安全配置
    ENABLE_RBAC = os.environ.get("ENABLE_RBAC", "true").lower() == "true"  # 启用RBAC权限控制
    ENABLE_AUDIT_LOG = os.environ.get("ENABLE_AUDIT_LOG", "true").lower() == "true"  # 启用审计日志
    
    # 容错配置
    ENABLE_FALLBACK = os.environ.get("ENABLE_FALLBACK", "true").lower() == "true"  # 启用故障降级
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """获取配置字典"""
        return {
            "base_dir": cls.BASE_DIR,
            "doc_path": cls.DOC_PATH,
            "use_domestic_stack": cls.USE_DOMESTIC_STACK,
            "use_redis_cache": cls.USE_REDIS_CACHE,
            "use_multiprocess": cls.USE_MULTIPROCESS,
            "env_mode": cls.ENV_MODE,
            "embed_model_name": cls.EMBED_MODEL_NAME,
            "embed_dim": cls.EMBED_DIM,
            "dp_epsilon": cls.DP_EPSILON,
            "dp_delta": cls.DP_DELTA,
            "similarity_threshold": cls.SIMILARITY_THRESHOLD,
            "top_k": cls.TOP_K,
            "max_context_size": cls.MAX_CONTEXT_SIZE,
            "local_qwen2_path": cls.LOCAL_QWEN2_PATH,
            "glm4_api_key": cls.GLM4_API_KEY,
            "glm4_api_url": cls.GLM4_API_URL,
            "qwen_api_key": cls.QWEN_API_KEY,
            "qwen_api_url": cls.QWEN_API_URL,
            "opensearch_host": cls.OPENSEARCH_HOST,
            "opensearch_port": cls.OPENSEARCH_PORT,
            "opensearch_user": cls.OPENSEARCH_USER,
            "opensearch_password": cls.OPENSEARCH_PASSWORD,
            "opensearch_index": cls.OPENSEARCH_INDEX,
            "redis_host": cls.REDIS_HOST,
            "redis_port": cls.REDIS_PORT,
            "redis_password": cls.REDIS_PASSWORD,
            "redis_db": cls.REDIS_DB,
            "redis_ttl": cls.REDIS_TTL,
            "worker_count": cls.WORKER_COUNT,
            "enable_rbac": cls.ENABLE_RBAC,
            "enable_audit_log": cls.ENABLE_AUDIT_LOG,
            "enable_fallback": cls.ENABLE_FALLBACK,
        }


# 导出配置实例
config = Config()
