# ==============================================
# 企业级石化RAG系统 - 大模型接口
# 核心功能：抽象大模型接口，支持国产大模型
# ==============================================

import torch
import requests
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from rag_dp_llm.config import config


class LLMInterface:
    """大模型抽象接口"""
    
    def generate_answer(self, input_dict: dict) -> str:
        """生成回答"""
        raise NotImplementedError


class Qwen2LLM(LLMInterface):
    """Qwen2-0.5B模型实现"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or config.LOCAL_QWEN2_PATH
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, padding_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
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
严格基于参考信息回答问题，分点清晰、数值准确，无相关信息时回答"无相关记录"。
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


class GLM4LLM(LLMInterface):
    """智谱清言GLM-4模型实现"""
    
    def __init__(self, api_key: str = None, api_url: str = None):
        self.api_key = api_key or config.GLM4_API_KEY
        self.api_url = api_url or config.GLM4_API_URL
        
        if not self.api_key:
            raise ValueError("GLM-4 API密钥未配置")
        
        print("✅ GLM-4模型接口初始化完成")
    
    def generate_answer(self, input_dict: dict) -> str:
        context = input_dict.get("context", "")
        question = input_dict.get("input", "")
        
        messages = [
            {
                "role": "system",
                "content": "严格基于参考信息回答问题，分点清晰、数值准确，无相关信息时回答\"无相关记录\"。"
            },
            {
                "role": "user",
                "content": f"参考信息：\n{context}\n\n问题：{question}"
            }
        ]
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": "glm-4",
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.0
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()
            
            return answer if answer and "无相关信息" not in answer else "无相关记录"
        except Exception as e:
            print(f"❌ GLM-4模型调用失败: {e}")
            return "无相关记录"


class QwenLLM(LLMInterface):
    """通义千问模型实现"""
    
    def __init__(self, api_key: str = None, api_url: str = None):
        self.api_key = api_key or config.QWEN_API_KEY
        self.api_url = api_url or config.QWEN_API_URL
        
        if not self.api_key:
            raise ValueError("通义千问API密钥未配置")
        
        print("✅ 通义千问模型接口初始化完成")
    
    def generate_answer(self, input_dict: dict) -> str:
        context = input_dict.get("context", "")
        question = input_dict.get("input", "")
        
        messages = [
            {
                "role": "system",
                "content": "严格基于参考信息回答问题，分点清晰、数值准确，无相关信息时回答\"无相关记录\"。"
            },
            {
                "role": "user",
                "content": f"参考信息：\n{context}\n\n问题：{question}"
            }
        ]
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": "qwen-turbo",
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.0
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()
            
            return answer if answer and "无相关信息" not in answer else "无相关记录"
        except Exception as e:
            print(f"❌ 通义千问模型调用失败: {e}")
            return "无相关记录"


class LLMFactory:
    """大模型工厂"""
    
    @staticmethod
    def create_llm() -> LLMInterface:
        """创建大模型实例"""
        if config.USE_DOMESTIC_STACK:
            # 优先使用GLM-4
            if config.GLM4_API_KEY:
                return GLM4LLM()
            # 其次使用通义千问
            elif config.QWEN_API_KEY:
                return QwenLLM()
            else:
                raise ValueError("国产大模型API密钥未配置")
        else:
            # 使用原始Qwen2模型
            return Qwen2LLM()
