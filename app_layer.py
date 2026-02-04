# ==============================================
# 企业级石化RAG系统 - 应用层
# 核心功能：对外服务接口、生产运维端入口
# ==============================================

import time
import json
from typing import Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor

from rag_dp_llm.config import config
from rag_dp_llm.service_layer import ServiceManager


class RAGAPI:
    """RAG服务API接口"""
    
    def __init__(self):
        self.service_manager = ServiceManager()
        self.rag_service = self.service_manager.get_rag_service()
        self.fault_tolerance = self.service_manager.get_fault_tolerance_service()
        
        if config.USE_MULTIPROCESS:
            self.executor = ProcessPoolExecutor(max_workers=config.WORKER_COUNT)
            print(f"✅ 多进程服务初始化完成，工作进程数: {config.WORKER_COUNT}")
        else:
            self.executor = None
        
        print("✅ RAG API接口初始化完成")
    
    def _process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个请求"""
        query = request.get('query', '')
        user_info = request.get('user_info', {
            'user_id': 'anonymous',
            'role': 'operator',
            '车间': None
        })
        
        return self.rag_service.generate_answer(query, user_info)
    
    def query(self, query: str, user_id: str = 'anonymous', 
              role: str = 'operator', 车间: str = None) -> Dict[str, Any]:
        """同步查询接口"""
        user_info = {
            'user_id': user_id,
            'role': role,
            '车间': 车间
        }
        
        request = {
            'query': query,
            'user_info': user_info
        }
        
        return self._process_request(request)
    
    def async_query(self, query: str, user_id: str = 'anonymous', 
                   role: str = 'operator', 车间: str = None) -> Any:
        """异步查询接口"""
        if not self.executor:
            return self.query(query, user_id, role, 车间)
        
        user_info = {
            'user_id': user_id,
            'role': role,
            '车间': 车间
        }
        
        request = {
            'query': query,
            'user_info': user_info
        }
        
        return self.executor.submit(self._process_request, request)
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        health_status = self.fault_tolerance.get_health_status()
        
        return {
            'status': 'healthy' if all(health_status.values()) else 'degraded',
            'services': health_status,
            'timestamp': time.time()
        }
    
    def batch_query(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量查询接口"""
        if self.executor:
            futures = [self.executor.submit(self._process_request, req) for req in queries]
            return [future.result() for future in futures]
        else:
            return [self._process_request(req) for req in queries]


class ProductionOperationPortal:
    """生产运维端入口"""
    
    def __init__(self):
        self.rag_api = RAGAPI()
        print("✅ 生产运维端入口初始化完成")
    
    def run_interactive(self):
        """交互式运行"""
        print("\n========== 石化生产运维RAG问答系统 ==========")
        print("系统已启动，输入问题进行咨询，输入'quit'退出")
        print("======================================")
        
        while True:
            try:
                question = input("\n请输入问题: ").strip()
                
                if question.lower() == 'quit':
                    print("系统已退出")
                    break
                
                if not question:
                    continue
                
                # 默认用户信息
                user_info = {
                    'user_id': 'operator_001',
                    'role': 'operator',
                    '车间': '炼油车间'
                }
                
                print("🤖 系统正在处理...")
                result = self.rag_api.query(question, **user_info)
                
                print(f"\n🤖 回答: {result['answer']}")
                print(f"📊 状态: {result['status']}")
                if 'response_time' in result:
                    print(f"⏱️  响应时间: {result['response_time']}秒")
                
            except KeyboardInterrupt:
                print("\n系统已退出")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")
    
    def run_test(self):
        """运行测试"""
        test_questions = [
            "天然气有哪些用途？",
            "金陵石化350万吨炼化装置的核心工艺是什么？",
            "合成氨的反应温度是多少？",
            "原油裂化的反应压力是多少？"
        ]
        
        print("\n========== 系统测试 ==========")
        
        for idx, question in enumerate(test_questions, 1):
            print(f"\n📝 问题{idx}：{question}")
            
            user_info = {
                'user_id': f'test_user_{idx}',
                'role': 'operator',
                '车间': '炼油车间'
            }
            
            result = self.rag_api.query(question, **user_info)
            print(f"🤖 回答：{result['answer']}")
            print(f"📊 状态：{result['status']}")
            if 'response_time' in result:
                print(f"⏱️  响应时间：{result['response_time']}秒")
        
        print("\n========== 测试完成 ==========")


class APIServer:
    """API服务器（简化版）"""
    
    def __init__(self):
        self.rag_api = RAGAPI()
        print("✅ API服务器初始化完成")
    
    def handle_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理HTTP请求"""
        try:
            if request_data.get('action') == 'query':
                return self.rag_api.query(
                    query=request_data.get('query', ''),
                    user_id=request_data.get('user_id', 'anonymous'),
                    role=request_data.get('role', 'operator'),
                    车间=request_data.get('车间', None)
                )
            
            elif request_data.get('action') == 'health_check':
                return self.rag_api.get_health_status()
            
            elif request_data.get('action') == 'batch_query':
                queries = request_data.get('queries', [])
                return self.rag_api.batch_query(queries)
            
            else:
                return {
                    'status': 'error',
                    'error': '未知操作'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def run_demo_server(self):
        """运行演示服务器"""
        print("\n========== API服务器演示 ==========")
        print("输入JSON格式的请求，输入'quit'退出")
        print("示例请求：{\"action\": \"query\", \"query\": \"天然气有哪些用途？\"}")
        print("=================================")
        
        while True:
            try:
                input_str = input("\n请输入请求: ").strip()
                
                if input_str.lower() == 'quit':
                    print("服务器已停止")
                    break
                
                if not input_str:
                    continue
                
                request_data = json.loads(input_str)
                result = self.handle_request(request_data)
                
                print(f"\n📡 响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
                
            except json.JSONDecodeError:
                print("❌ JSON格式错误，请重新输入")
            except KeyboardInterrupt:
                print("\n服务器已停止")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")
