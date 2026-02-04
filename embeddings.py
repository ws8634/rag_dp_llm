# ==============================================
# 企业级石化RAG系统 - 嵌入层
# 核心功能：实现带差分隐私的语义嵌入
# ==============================================

import numpy as np
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings

from rag_dp_llm.config import config


class DP_SemanticEmbeddings:
    """带差分隐私的语义嵌入模型"""
    
    def __init__(self, embed_model_name: str = None, epsilon: float = None, delta: float = None):
        """
        初始化嵌入模型
        
        Args:
            embed_model_name: 嵌入模型名称
            epsilon: 差分隐私ε值
            delta: 差分隐私δ值
        """
        self.embed_model_name = embed_model_name or config.EMBED_MODEL_NAME
        self.epsilon = epsilon or config.DP_EPSILON
        self.delta = delta or config.DP_DELTA
        
        # 初始化基础嵌入模型
        self.base_embeddings = HuggingFaceEmbeddings(
            model_name=self.embed_model_name,
            model_kwargs={"device": "cpu"},  # 强制CPU，适配低配置
            encode_kwargs={"normalize_embeddings": True}  # 归一化向量
        )
        
        self.embed_dim = config.EMBED_DIM
        self.sigma = np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon  # 计算噪声标准差
        np.random.seed(42)  # 固定种子，保证检索一致性
        
        print(f"✅ 差分隐私嵌入模型初始化完成（ε={self.epsilon}）")
    
    def _add_dp_noise(self, vector: List[float]) -> List[float]:
        """
        添加差分隐私噪声
        
        Args:
            vector: 原始向量
            
        Returns:
            添加噪声后的向量
        """
        noise = np.random.normal(0, self.sigma, len(vector))
        noisy_vector = [float(v + n) for v, n in zip(vector, noise)]
        norm = np.linalg.norm(noisy_vector)
        noisy_vector = [v / norm for v in noisy_vector]
        return noisy_vector
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        嵌入文档列表
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
        """
        base_vecs = self.base_embeddings.embed_documents(texts)
        return [self._add_dp_noise(vec) for vec in base_vecs]
    
    def embed_query(self, text: str) -> List[float]:
        """
        嵌入查询文本
        
        Args:
            text: 查询文本
            
        Returns:
            嵌入向量
        """
        base_vec = self.base_embeddings.embed_query(text)
        return self._add_dp_noise(base_vec)
