# ==============================================
# 企业级石化RAG系统主入口
# 核心功能：整合所有模块，提供完整的RAG服务
# ==============================================

import os
import sys
import argparse
from typing import Dict, Any

from rag_dp_llm.config import config
from rag_dp_llm.app_layer import ProductionOperationPortal, APIServer
from rag_dp_llm.service_layer import ServiceManager


def print_system_info():
    """打印系统信息"""
    print("\n========================================")
    print("企业级石化生产运维RAG问答系统")
    print("========================================")
    print(f"环境模式: {config.ENV_MODE}")
    print(f"国产化技术栈: {'启用' if config.USE_DOMESTIC_STACK else '禁用'}")
    print(f"Redis缓存: {'启用' if config.USE_REDIS_CACHE else '禁用'}")
    print(f"多进程服务: {'启用' if config.USE_MULTIPROCESS else '禁用'}")
    if config.USE_MULTIPROCESS:
        print(f"工作进程数: {config.WORKER_COUNT}")
    print("========================================")


def run_production_portal():
    """运行生产运维端"""
    portal = ProductionOperationPortal()
    portal.run_interactive()


def run_test():
    """运行系统测试"""
    portal = ProductionOperationPortal()
    portal.run_test()


def run_api_server():
    """运行API服务器"""
    server = APIServer()
    server.run_demo_server()


def show_health_status():
    """显示健康状态"""
    service_manager = ServiceManager()
    fault_tolerance = service_manager.get_fault_tolerance_service()
    health_status = fault_tolerance.get_health_status()
    
    print("\n========================================")
    print("系统健康状态")
    print("========================================")
    for service, status in health_status.items():
        print(f"{service}: {'✅ 正常' if status else '❌ 异常'}")
    print("========================================")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='企业级石化RAG系统')
    parser.add_argument('--mode', type=str, default='portal', 
                      choices=['portal', 'test', 'api', 'health'],
                      help='运行模式: portal(生产运维端), test(系统测试), api(API服务器), health(健康检查)')
    
    args = parser.parse_args()
    
    # 打印系统信息
    print_system_info()
    
    # 根据模式运行
    if args.mode == 'portal':
        run_production_portal()
    elif args.mode == 'test':
        run_test()
    elif args.mode == 'api':
        run_api_server()
    elif args.mode == 'health':
        show_health_status()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 系统运行出错: {str(e)[:800]}")
        import traceback
        traceback.print_exc()
