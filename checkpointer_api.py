from checkpointer import *
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Depends, Query, Path, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime

class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None


class SaveConversationRequest(BaseModel):
    thread_id: str = Field(..., description="线程ID")
    conversation_id: str = Field(..., description="会话ID")
    messages: List[Message] = Field(..., description="消息列表")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="元数据")


class SaveConversationResponse(BaseModel):
    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None


class GetConversationResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class DeleteConversationResponse(BaseModel):
    success: bool
    deleted: bool
    error: Optional[str] = None


class GetConversationsResponse(BaseModel):
    success: bool
    count: int
    data: List[Dict[str, Any]]
    error: Optional[str] = None


app = FastAPI(
    title="聊天历史管理 API",
    description="提供聊天历史的保存、查询和删除功能",
    version="1.0.0"
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请指定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局 ChatManager 实例
_chat_manager: Optional[ChatManager] = None


async def get_chat_manager() -> ChatManager:
    """获取 ChatManager 实例"""
    global _chat_manager
    if _chat_manager is None:
        _chat_manager = ChatManager()
        await _chat_manager.initialize()
    return _chat_manager


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    await get_chat_manager()


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    global _chat_manager
    if _chat_manager:
        await _chat_manager.close()
        _chat_manager = None


# ====================== API 路由 ======================
@app.post("/api/v1/conversations",
          response_model=SaveConversationResponse,
          summary="保存对话",
          description="保存或更新对话消息")
async def save_conversation(
        request: SaveConversationRequest,
        manager: ChatManager = Depends(get_chat_manager)
):
    try:
        # 转换 Pydantic 模型为字典列表
        messages_dict = [msg.dict() for msg in request.messages]

        message_id = await manager.save_conversation(
            thread_id=request.thread_id,
            conversation_id=request.conversation_id,
            messages=messages_dict,
            metadata=request.metadata
        )

        return SaveConversationResponse(
            success=True,
            message_id=message_id
        )
    except Exception as e:
        logging.error(f"保存对话失败: {e}")
        return SaveConversationResponse(
            success=False,
            error=str(e)
        )


@app.get("/api/v1/conversations/{thread_id}/{conversation_id}",
         response_model=GetConversationResponse,
         summary="获取对话",
         description="根据线程ID和会话ID获取对话详情")
async def get_conversation(
        thread_id: str = Path(..., description="线程ID"),
        conversation_id: str = Path(..., description="会话ID"),
        manager: ChatManager = Depends(get_chat_manager)
):
    try:
        conversation = await manager.get_conversation(thread_id, conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="对话不存在")

        return GetConversationResponse(
            success=True,
            data=conversation
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"获取对话失败: {e}")
        return GetConversationResponse(
            success=False,
            error=str(e)
        )


@app.delete("/api/v1/conversations/{thread_id}/{conversation_id}",
            response_model=DeleteConversationResponse,
            summary="删除对话",
            description="删除指定的对话")
async def delete_conversation(
        thread_id: str = Path(..., description="线程ID"),
        conversation_id: str = Path(..., description="会话ID"),
        manager: ChatManager = Depends(get_chat_manager)
):
    try:
        deleted = await manager.delete_conversation(thread_id, conversation_id)
        return DeleteConversationResponse(
            success=True,
            deleted=deleted
        )
    except Exception as e:
        logging.error(f"删除对话失败: {e}")
        return DeleteConversationResponse(
            success=False,
            deleted=False,
            error=str(e)
        )


@app.get("/api/v1/conversations/today",
         response_model=GetConversationsResponse,
         summary="获取今日对话",
         description="获取今日的所有对话")
async def get_todays_conversations(
        thread_id: str = Query(..., description="线程ID（可选）"),
        limit: int = Query(100, description="返回数量限制", ge=1, le=1000),
        manager: ChatManager = Depends(get_chat_manager)
):
    try:
        conversations = await manager.get_todays_conversations(thread_id, limit)
        return GetConversationsResponse(
            success=True,
            count=len(conversations),
            data=conversations
        )
    except Exception as e:
        logging.error(f"获取今日对话失败: {e}")
        return GetConversationsResponse(
            success=False,
            count=0,
            data=[],
            error=str(e)
        )


@app.get("/api/v1/conversations/yesterday",
         response_model=GetConversationsResponse,
         summary="获取昨日对话",
         description="获取昨日的所有对话")
async def get_yesterday_conversations(
        thread_id: str = Query(..., description="线程ID（可选）"),
        limit: int = Query(100, description="返回数量限制", ge=1, le=1000),
        manager: ChatManager = Depends(get_chat_manager)
):
    try:
        conversations = await manager.get_yesterday_conversations(thread_id, limit)
        return GetConversationsResponse(
            success=True,
            count=len(conversations),
            data=conversations
        )
    except Exception as e:
        logging.error(f"获取昨日对话失败: {e}")
        return GetConversationsResponse(
            success=False,
            count=0,
            data=[],
            error=str(e)
        )


@app.get("/api/v1/conversations/last-7days",
         response_model=GetConversationsResponse,
         summary="获取最近7天对话",
         description="获取最近7天（排除今日和昨日）的对话")
async def get_last_7days_conversations(
        thread_id: str = Query(..., description="线程ID（可选）"),
        limit: int = Query(100, description="返回数量限制", ge=1, le=1000),
        manager: ChatManager = Depends(get_chat_manager)
):
    try:
        conversations = await manager.get_last_7days_conversations(thread_id, limit)
        return GetConversationsResponse(
            success=True,
            count=len(conversations),
            data=conversations
        )
    except Exception as e:
        logging.error(f"获取最近7天对话失败: {e}")
        return GetConversationsResponse(
            success=False,
            count=0,
            data=[],
            error=str(e)
        )


@app.get("/api/v1/conversations/last-30days",
         response_model=GetConversationsResponse,
         summary="获取最近30天对话",
         description="获取最近30天（排除最近1天）的对话")
async def get_last_30days_conversations(
        thread_id: str = Query(..., description="线程ID（可选）"),
        limit: int = Query(100, description="返回数量限制", ge=1, le=1000),
        manager: ChatManager = Depends(get_chat_manager)
):
    try:
        conversations = await manager.get_last_30days_conversations(thread_id, limit)
        return GetConversationsResponse(
            success=True,
            count=len(conversations),
            data=conversations
        )
    except Exception as e:
        logging.error(f"获取最近30天对话失败: {e}")
        return GetConversationsResponse(
            success=False,
            count=0,
            data=[],
            error=str(e)
        )


@app.get("/api/v1/conversations/last-quarter",
         response_model=GetConversationsResponse,
         summary="获取最近季度对话",
         description="获取最近一个季度（约120天，排除最近30天）的对话")
async def get_last_quarter_conversations(
        thread_id: str = Query(..., description="线程ID（可选）"),
        limit: int = Query(100, description="返回数量限制", ge=1, le=1000),
        manager: ChatManager = Depends(get_chat_manager)
):
    try:
        conversations = await manager.get_last_quarter_conversations(thread_id, limit)
        return GetConversationsResponse(
            success=True,
            count=len(conversations),
            data=conversations
        )
    except Exception as e:
        logging.error(f"获取季度对话失败: {e}")
        return GetConversationsResponse(
            success=False,
            count=0,
            data=[],
            error=str(e)
        )


@app.get("/api/v1/conversations/last-half-year",
         response_model=GetConversationsResponse,
         summary="获取最近半年对话",
         description="获取最近半年（约180天，排除最近120天）的对话")
async def get_last_half_year_conversations(
        thread_id: str = Query(..., description="线程ID（可选）"),
        limit: int = Query(100, description="返回数量限制", ge=1, le=1000),
        manager: ChatManager = Depends(get_chat_manager)
):
    try:
        conversations = await manager.get_last_half_year_conversations(thread_id, limit)
        return GetConversationsResponse(
            success=True,
            count=len(conversations),
            data=conversations
        )
    except Exception as e:
        logging.error(f"获取半年对话失败: {e}")
        return GetConversationsResponse(
            success=False,
            count=0,
            data=[],
            error=str(e)
        )


@app.get("/api/v1/conversations/last-year",
         response_model=GetConversationsResponse,
         summary="获取最近一年对话",
         description="获取最近一年（约365天，排除最近180天）的对话")
async def get_last_year_conversations(
        thread_id: str = Query(..., description="线程ID（可选）"),
        limit: int = Query(100, description="返回数量限制", ge=1, le=1000),
        manager: ChatManager = Depends(get_chat_manager)
):
    try:
        conversations = await manager.get_last_year_conversations(thread_id, limit)
        return GetConversationsResponse(
            success=True,
            count=len(conversations),
            data=conversations
        )
    except Exception as e:
        logging.error(f"获取一年对话失败: {e}")
        return GetConversationsResponse(
            success=False,
            count=0,
            data=[],
            error=str(e)
        )


@app.get("/health", summary="健康检查", description="检查服务是否正常运行")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/", summary="API根路径", description="API基本信息")
async def root():
    return {
        "name": "聊天历史管理 API",
        "version": "1.0.0",
        "description": "提供聊天历史的保存、查询和删除功能",
        "endpoints": {
            "保存对话": "POST /api/v1/conversations",
            "获取对话": "GET /api/v1/conversations/{thread_id}/{conversation_id}",
            "删除对话": "DELETE /api/v1/conversations/{thread_id}/{conversation_id}",
            "获取今日对话": "GET /api/v1/conversations/today",
            "获取昨日对话": "GET /api/v1/conversations/yesterday",
            "获取最近7天对话": "GET /api/v1/conversations/last-7days",
            "获取最近30天对话": "GET /api/v1/conversations/last-30days",
            "获取最近季度对话": "GET /api/v1/conversations/last-quarter",
            "获取最近半年对话": "GET /api/v1/conversations/last-half-year",
            "获取最近一年对话": "GET /api/v1/conversations/last-year"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "checkpointer_api:app",
        host="0.0.0.0",
        port=8010,
        reload=True,
        log_level="info"
    )