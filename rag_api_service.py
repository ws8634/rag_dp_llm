# ==============================================
# RAG检索服务API接口
# 提供RESTful接口，供其他服务调用RAG检索功能
# ==============================================
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
import time
import json
import threading

from enterprise_rag_with_dp_opensearch import SystemConfig, load_and_split_docs, build_vector_db, LLMWrapper, RedisCache, RAGService

# 全局变量
app = FastAPI(
    title="石化生产运维RAG检索服务API",
    description="企业级石化生产运维知识库检索服务，支持国产化技术栈",
    version="1.0.0"
)
rag_service = None
vector_db = None
llm = None
redis_cache = None

# 请求模型
class RAGQueryRequest(BaseModel):
    query: str
    username: str = "anonymous"
    workshop: Optional[str] = None

# 响应模型
class RAGQueryResponse(BaseModel):
    query: str
    response: str
    username: str
    workshop: Optional[str]
    time_taken: float
    timestamp: str
    status: str

# 服务健康检查
class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    services: dict

# 初始化服务
@app.on_event("startup")
async def startup_event():
    global rag_service, vector_db, llm, redis_cache
    print("🚀 启动RAG服务API...")
    
    # 加载文档
    split_docs = load_and_split_docs(SystemConfig.DOC_PATH)
    
    # 构建向量库
    vector_db = build_vector_db(split_docs)
    
    # 初始化大模型
    llm = LLMWrapper()
    
    # 初始化Redis缓存
    redis_cache = RedisCache() if SystemConfig.USE_REDIS_CACHE else None
    
    # 初始化RAG服务
    rag_service = RAGService(vector_db, llm, redis_cache)
    
    print("✅ RAG服务API启动完成")

# 健康检查接口
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    from enterprise_rag_with_dp_opensearch import FaultTolerance
    fault_tolerance = FaultTolerance()
    
    services_status = {
        "vector_store": fault_tolerance.check_service_health("vector_store"),
        "llm": fault_tolerance.check_service_health("llm"),
        "redis": redis_cache.client is not None if redis_cache else False
    }
    
    return HealthCheckResponse(
        status="healthy" if all(services_status.values()) else "degraded",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        services=services_status
    )

# RAG检索接口
@app.post("/api/rag/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG服务未初始化")
    
    start_time = time.time()
    
    try:
        # 处理查询
        response = rag_service.process_query(
            query=request.query,
            username=request.username,
            workshop=request.workshop
        )
        
        end_time = time.time()
        time_taken = end_time - start_time
        
        return RAGQueryResponse(
            query=request.query,
            response=response,
            username=request.username,
            workshop=request.workshop,
            time_taken=time_taken,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            status="success"
        )
    except Exception as e:
        end_time = time.time()
        time_taken = end_time - start_time
        
        return RAGQueryResponse(
            query=request.query,
            response=f"处理失败：{str(e)}",
            username=request.username,
            workshop=request.workshop,
            time_taken=time_taken,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            status="error"
        )

# 批量查询接口
class BatchQueryItem(BaseModel):
    id: str
    query: str
    username: str = "anonymous"
    workshop: Optional[str] = None

class BatchQueryRequest(BaseModel):
    queries: list[BatchQueryItem]

class BatchQueryResponse(BaseModel):
    results: list[dict]
    total: int
    timestamp: str

@app.post("/api/rag/batch_query", response_model=BatchQueryResponse)
async def batch_query(request: BatchQueryRequest):
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG服务未初始化")
    
    results = []
    
    for item in request.queries:
        try:
            response = rag_service.process_query(
                query=item.query,
                username=item.username,
                workshop=item.workshop
            )
            results.append({
                "id": item.id,
                "query": item.query,
                "response": response,
                "status": "success"
            })
        except Exception as e:
            results.append({
                "id": item.id,
                "query": item.query,
                "response": f"处理失败：{str(e)}",
                "status": "error"
            })
    
    return BatchQueryResponse(
        results=results,
        total=len(results),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )

# 系统配置接口
@app.get("/api/config")
async def get_config():
    return {
        "run_mode": SystemConfig.RUN_MODE,
        "use_domestic_stack": SystemConfig.USE_DOMESTIC_STACK,
        "use_redis_cache": SystemConfig.USE_REDIS_CACHE,
        "use_multiprocess": SystemConfig.USE_MULTIPROCESS,
        "process_count": SystemConfig.PROCESS_COUNT,
        "embedding_model": SystemConfig.EMBED_MODEL_NAME,
        "similarity_threshold": SystemConfig.SIMILARITY_THRESHOLD,
        "dp_epsilon": SystemConfig.DP_EPSILON
    }

# 启动服务
if __name__ == "__main__":
    # 修改配置为服务器模式
    SystemConfig.RUN_MODE = "server"
    SystemConfig.USE_REDIS_CACHE = True
    
    # 启动服务
    uvicorn.run(
        "rag_api_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True if SystemConfig.RUN_MODE == "development" else False
    )