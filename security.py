# ==============================================
# 企业级石化RAG系统 - 安全模块
# 核心功能：RBAC权限控制、审计日志
# ==============================================

import time
import json
import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

from rag_dp_llm.config import config


class RBACManager:
    """基于RBAC的权限管理器"""
    
    def __init__(self):
        # 角色权限映射
        self.role_permissions = {
            'admin': ['query_rag', 'manage_users', 'manage_docs', 'view_all'],
            'operator': ['query_rag', 'view_own_workshop'],
            'manager': ['query_rag', 'view_department', 'manage_workshop']
        }
        
        # 角色继承关系
        self.role_hierarchy = {
            'operator': [],
            'manager': ['operator'],
            'admin': ['manager', 'operator']
        }
        
        print("✅ RBAC权限管理器初始化完成")
    
    def check_permission(self, role: str, permission: str) -> bool:
        """检查权限"""
        if role not in self.role_permissions:
            return False
        
        # 检查直接权限
        if permission in self.role_permissions[role]:
            return True
        
        # 检查继承权限
        for parent_role in self.role_hierarchy.get(role, []):
            if self.check_permission(parent_role, permission):
                return True
        
        return False
    
    def get_role_permissions(self, role: str) -> List[str]:
        """获取角色权限"""
        permissions = set()
        
        def collect_permissions(r):
            permissions.update(self.role_permissions.get(r, []))
            for parent_role in self.role_hierarchy.get(r, []):
                collect_permissions(parent_role)
        
        collect_permissions(role)
        return list(permissions)
    
    def filter_documents(self, documents: List[Document], role: str, 车间: str = None) -> List[Document]:
        """根据权限过滤文档"""
        if role == 'admin':
            return documents
        
        filtered_docs = []
        for doc in documents:
            doc_workshop = doc.metadata.get('车间', '通用')
            
            if role == 'manager':
                # 管理者可以查看本部门文档
                if doc_workshop == 车间 or doc_workshop == '通用':
                    filtered_docs.append(doc)
            elif role == 'operator':
                # 操作员只能查看本车间文档
                if doc_workshop == 车间 or doc_workshop == '通用':
                    filtered_docs.append(doc)
        
        return filtered_docs


class AuditLogger:
    """审计日志记录器"""
    
    def __init__(self):
        self.log_dir = os.path.join(config.BASE_DIR, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.audit_log_file = os.path.join(self.log_dir, "audit.log")
        self.error_log_file = os.path.join(self.log_dir, "error.log")
        print("✅ 审计日志记录器初始化完成")
    
    def _write_log(self, log_file: str, log_entry: Dict[str, Any]):
        """写入日志"""
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def log_access(self, user_id: str, action: str, resource: str, details: Dict[str, Any]):
        """记录访问日志"""
        log_entry = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'details': details,
            'status': 'success'
        }
        
        self._write_log(self.audit_log_file, log_entry)
        print(f"📝 审计日志: {user_id} 执行 {action} 操作")
    
    def log_error(self, user_id: str, action: str, error: str):
        """记录错误日志"""
        log_entry = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'user_id': user_id,
            'action': action,
            'error': error,
            'status': 'error'
        }
        
        self._write_log(self.error_log_file, log_entry)
        self._write_log(self.audit_log_file, log_entry)
        print(f"📝 错误日志: {user_id} 执行 {action} 操作失败: {error}")
    
    def log_model_call(self, user_id: str, model_name: str, input_text: str, output_text: str, duration: float):
        """记录模型调用日志"""
        log_entry = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'user_id': user_id,
            'action': 'model_call',
            'model_name': model_name,
            'input_length': len(input_text),
            'output_length': len(output_text),
            'duration': duration,
            'status': 'success'
        }
        
        self._write_log(self.audit_log_file, log_entry)
        print(f"📝 模型调用日志: {user_id} 调用 {model_name}，耗时 {duration:.2f}s")
    
    def get_recent_logs(self, hours: int = 24) -> List[Dict[str, Any]]:
        """获取最近的日志"""
        logs = []
        cutoff_time = time.time() - (hours * 3600)
        
        try:
            with open(self.audit_log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)
                        log_time = time.mktime(time.strptime(
                            log_entry['timestamp'], '%Y-%m-%d %H:%M:%S'
                        ))
                        if log_time >= cutoff_time:
                            logs.append(log_entry)
                    except Exception:
                        pass
        except Exception:
            pass
        
        return logs[-100:]  # 返回最近100条


class SecurityManager:
    """安全管理器"""
    
    _instance = None
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.rbac_manager = RBACManager()
            self.audit_logger = AuditLogger()
            self._initialized = True
    
    def get_rbac_manager(self) -> RBACManager:
        return self.rbac_manager
    
    def get_audit_logger(self) -> AuditLogger:
        return self.audit_logger
