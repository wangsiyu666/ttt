"""
用户对话历史
"""
import logging
import json
import asyncio
from typing import (
    Optional,
    List,
    Tuple, Any, Dict, AsyncGenerator
)

import asyncpg
from contextlib import asynccontextmanager
import os
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class DatabaseConfig:
    host: str = os.getenv("DB_HOST", "192.168.124.137")
    port: int = int(os.getenv("DB_PORT", "5432"))
    database: str = os.getenv("DB_DATABASE", "postgres")
    user: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASSWORD", "123456")
    min_size: int = int(os.getenv("DB_POOL_MIN_SIZE", "4"))
    max_size: int = int(os.getenv("DB_POOL_MAX_SIZE", "10"))


class BaseRepository(ABC):
    """仓储基类"""

    @abstractmethod
    async def initialize(self) -> None:
        """初始化资源"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """关闭资源"""
        pass

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class ConnectionPoolManager:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._pool: Optional[asyncpg.Pool] = None
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger(f"{__name__}.ConnectionPoolManager")

    async def get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            async with self._lock:
                if self._pool is None:
                    self.logger.info(f"Creating connection pool to {self.config.host}: {self.config.port}")
                    self._pool = await asyncpg.create_pool(
                        host=self.config.host,
                        port=self.config.port,
                        database=self.config.database,
                        user=self.config.user,
                        password=self.config.password,
                        min_size=self.config.min_size,
                        max_size=self.config.max_size,
                    )
        return self._pool

    async def close(self):
        if self._pool:
            self.logger.info(f"Closing connection pool")
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        pool = await self.get_pool()
        conn = await pool.acquire()
        try:
            yield conn
        finally:
            await pool.release(conn)

    @asynccontextmanager
    async def get_transaction(self) -> AsyncGenerator[asyncpg.Connection, None]:
        pool = await self.get_pool()
        conn = await pool.acquire()
        try:
            async with conn.transaction():
                yield conn
        finally:
            await pool.release(conn)


class ChatHistoryRepository(BaseRepository):
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._pool_manager: Optional[ConnectionPoolManager] = None
        self.logger = logging.getLogger(f"{__name__}.ChatHistoryRepository")
        self._initialized = False

    async def initialize(self) -> None:
        if not self._initialized:
            self._pool_manager = ConnectionPoolManager(self.config)
            await self._create_tables()
            self._initialized = True
            self.logger.info("Chat history repository initialized")

    async def _create_tables(self):
        async with self._pool_manager.get_connection() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    thread_id VARCHAR(255) NOT NULL,
                    conversation_id VARCHAR(255) NOT NULL,
                    messages JSONB NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    version INTEGER DEFAULT 1
                );
                
                CREATE INDEX IF NOT EXISTS idx_chat_history_thread 
                ON chat_history(thread_id);
                
                CREATE INDEX IF NOT EXISTS idx_chat_history_conversation 
                ON chat_history(conversation_id);
                
                CREATE INDEX IF NOT EXISTS idx_chat_history_composite 
                ON chat_history(thread_id, conversation_id);
                
                CREATE INDEX IF NOT EXISTS idx_chat_history_updated 
                ON chat_history(updated_at);
                
                CREATE INDEX IF NOT EXISTS idx_chat_history_messages 
                ON chat_history USING GIN (messages);
                """
            )
            self.logger.info("Chat history table created")

    async def save_messages(
            self,
            thread_id: str,
            conversation_id: str,
            messages: List[Dict[str, Any]],
            metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if metadata is None:
            metadata = {}

        try:
            async with self._pool_manager.get_connection() as conn:
                record = await conn.fetchrow(
                    """
                    INSERT INTO chat_history
                        (thread_id, conversation_id, messages, metadata, version)
                    VALUES ($1, $2, $3, $4, 1) RETURNING id, version
                    """,
                    thread_id, conversation_id, json.dumps(messages), json.dumps(metadata)
                )
                if record:
                    self.logger.debug(
                        f"Saved messages for thread={thread_id}, "
                        f"conversation={conversation_id}, version={record['version']}"
                    )
                    return str(record['id'])
        except asyncpg.PostgresError as e:
            self.logger.error(f"Database error saving messages: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error saving messages: {e}")
            raise
        return None

    async def get_messages(
            self,
            thread_id: str,
            conversation_id: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            async with self._pool_manager.get_connection() as conn:
                record = await conn.fetchrow(
                    """
                    SELECT
                        id,
                        thread_id,
                        conversation_id,
                        messages,
                        metadata,
                        created_at,
                        updated_at,
                        version
                    FROM chat_history 
                    WHERE thread_id = $1 AND conversation_id = $2
                    """,
                    thread_id, conversation_id
                )
                if record:
                    return {
                        'id': str(record['id']),
                        'thread_id': record['thread_id'],
                        'conversation_id': record['conversation_id'],
                        'messages': record['messages'],
                        'metadata': record['metadata'],
                        'created_at': record['created_at'],
                        'updated_at': record['updated_at'],
                        'version': record['version']
                    }
        except asyncpg.PostgresError as e:
            self.logger.error(f"Database error fetching messages: {e}")
            raise

        return None

    async def delete_messages(
            self,
            thread_id: str,
            conversation_id: str
    ) -> bool:
        """删除聊天消息"""
        try:
            async with self._pool_manager.get_transaction() as conn:
                result = await conn.execute("""
                                            DELETE
                                            FROM chat_history
                                            WHERE thread_id = $1
                                              AND conversation_id = $2
                                            """, thread_id, conversation_id)

                deleted = "DELETE" in result
                if deleted:
                    self.logger.info(
                        f"Deleted messages for thread={thread_id}, "
                        f"conversation={conversation_id}"
                    )
                return deleted
        except asyncpg.PostgresError as e:
            self.logger.error(f"Database error deleting messages: {e}")
            raise

    async def get_todays_conversations(
            self,
            thread_id: Optional[str] = None,
            limit: int = 100
    ) -> List[Dict[str, Any]]:
        try:
            async with self._pool_manager.get_connection() as conn:
                query = """
                        SELECT ch.id, \
                               ch.thread_id, \
                               ch.conversation_id, \
                               ch.messages, \
                               ch.metadata, \
                               ch.created_at, \
                               ch.updated_at, \
                               ch.version
                        FROM chat_history ch
                        WHERE DATE (ch.created_at) = CURRENT_DATE
                          AND ($1::text IS NULL \
                           OR ch.thread_id = $1)
                        ORDER BY ch.created_at DESC
                            LIMIT $2 \
                        """
                records = await conn.fetch(query, thread_id, limit)
                return [
                    {
                        'id': str(record['id']),
                        'thread_id': record['thread_id'],
                        'conversation_id': record['conversation_id'],
                        'messages': record['messages'],
                        'metadata': record['metadata'],
                        'created_at': record['created_at'],
                        'updated_at': record['updated_at'],
                        'version': record['version']
                    }
                    for record in records
                ]
        except asyncpg.PostgresError as e:
            self.logger.error(f"Database error getting today's conversations: {e}")
            raise

    async def get_yesterday_conversations(
            self,
            thread_id: Optional[str] = None,
            limit: int = 100
    ) -> List[Dict[str, Any]]:
        try:
            async with self._pool_manager.get_connection() as conn:
                query = """
                SELECT
                        ch.id,
                        ch.thread_id,
                        ch.conversation_id,
                        ch.messages,
                        ch.metadata,
                        ch.created_at,
                        ch.updated_at,
                        ch.version
                    FROM chat_history ch
                    WHERE created_at::date = CURRENT_DATE - INTERVAL '1 day'
                    AND thread_id = $1
                    ORDER BY ch.created_at DESC
                    LIMIT $2
                """
                records = await conn.fetch(query, thread_id, limit)
                return [
                    {
                        'id': str(record['id']),
                        'thread_id': record['thread_id'],
                        'conversation_id': record['conversation_id'],
                        'messages': record['messages'],
                        'metadata': record['metadata'],
                        'created_at': record['created_at'],
                        'updated_at': record['updated_at'],
                        'version': record['version']
                    }
                    for record in records
                ]
        except asyncpg.PostgresError as e:
            self.logger.error(f"Database error getting yesterday's conversations: {e}")
            raise

    async def get_last_7days_conversations(
            self,
            thread_id: Optional[str] = None,
            limit: int = 100
    ) -> List[Dict[str, Any]]:
        try:
            async with self._pool_manager.get_connection() as conn:
                query = """
                    SELECT
                        ch.id,
                        ch.thread_id,
                        ch.conversation_id,
                        ch.messages,
                        ch.metadata,
                        ch.created_at,
                        ch.updated_at,
                        ch.version
                    FROM chat_history ch
                    WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
                    AND created_at < CURRENT_DATE - INTERVAL '1 day'
                    AND thread_id = $1
                    ORDER BY ch.created_at DESC
                    LIMIT $2
                """
                records = await conn.fetch(query, thread_id, limit)
                return [
                    {
                        'id': str(record['id']),
                        'thread_id': record['thread_id'],
                        'conversation_id': record['conversation_id'],
                        'messages': record['messages'],
                        'metadata': record['metadata'],
                        'created_at': record['created_at'],
                        'updated_at': record['updated_at'],
                        'version': record['version']
                    }
                    for record in records
                ]
        except asyncpg.PostgresError as e:
            self.logger.error(f"Database error getting last 7 days conversations (exclude today & yesterday): {e}")
            raise

    async def get_last_30days_conversations(
            self,
            thread_id: Optional[str] = None,
            limit: int = 100
    ) -> List[Dict[str, Any]]:
        try:
            async with self._pool_manager.get_connection() as conn:
                query = """
                SELECT
                    ch.id,
                    ch.thread_id,
                    ch.conversation_id,
                    ch.messages,
                    ch.metadata,
                    ch.created_at,
                    ch.updated_at,
                    ch.version
                FROM chat_history ch
                WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
                AND created_at < CURRENT_DATE - INTERVAL '7 days'
                AND thread_id = $1
                ORDER BY ch.created_at DESC
                LIMIT $2
                """
                records = await conn.fetch(query, thread_id, limit)
                return [
                    {
                        'id': str(record['id']),
                        'thread_id': record['thread_id'],
                        'conversation_id': record['conversation_id'],
                        'messages': record['messages'],
                        'metadata': record['metadata'],
                        'created_at': record['created_at'],
                        'updated_at': record['updated_at'],
                        'version': record['version']
                    }
                    for record in records
                ]

        except asyncpg.PostgresError as e:
            self.logger.error(f"Database error getting 30 days conversations: {e}")
            raise

    async def get_last_quarter_conversations(
            self,
            thread_id: Optional[str] = None,
            limit: int = 100
    ) -> List[Dict[str, Any]]:
        try:
            async with self._pool_manager.get_connection() as conn:
                query = """
                   SELECT
                        ch.id,
                        ch.thread_id,
                        ch.conversation_id,
                        ch.messages,
                        ch.metadata,
                        ch.created_at,
                        ch.updated_at,
                        ch.version
                    FROM chat_history ch
                    WHERE created_at >= CURRENT_DATE - INTERVAL '90 days'  -- 120天 = 4个月 ≈ 一个季度
                    AND created_at < CURRENT_DATE - INTERVAL '30 days'  -- 排除最近30天
                    AND thread_id = $1
                    ORDER BY ch.created_at DESC
                    LIMIT $2
                """
                records = await conn.fetch(query, thread_id, limit)
                return [
                    {
                        'id': str(record['id']),
                        'thread_id': record['thread_id'],
                        'conversation_id': record['conversation_id'],
                        'messages': record['messages'],
                        'metadata': record['metadata'],
                        'created_at': record['created_at'],
                        'updated_at': record['updated_at'],
                        'version': record['version']
                    }
                    for record in records
                ]
        except asyncpg.PostgresError as e:
            self.logger.error(f"Database error getting quarter conversations: {e}")
            raise

    async def get_last_half_year_conversations(
            self,
            thread_id: Optional[str] = None,
            limit: int = 100
    ) -> List[Dict[str, Any]]:
        try:
            async with self._pool_manager.get_connection() as conn:
                query = """
                   SELECT
                        ch.id,
                        ch.thread_id,
                        ch.conversation_id,
                        ch.messages,
                        ch.metadata,
                        ch.created_at,
                        ch.updated_at,
                        ch.version
                    FROM chat_history ch
                    WHERE created_at >= CURRENT_DATE - INTERVAL '180 days'  -- 180天 = 6个月
                    AND created_at < CURRENT_DATE - INTERVAL '120 days'  -- 排除最近120天
                    AND thread_id = $1
                    ORDER BY ch.created_at DESC
                    LIMIT $2
                """
                records = await conn.fetch(query, thread_id, limit)
                return [
                    {
                        'id': str(record['id']),
                        'thread_id': record['thread_id'],
                        'conversation_id': record['conversation_id'],
                        'messages': record['messages'],
                        'metadata': record['metadata'],
                        'created_at': record['created_at'],
                        'updated_at': record['updated_at'],
                        'version': record['version']
                    }
                    for record in records
                ]

        except asyncpg.PostgresError as e:
            self.logger.error(f"Database error getting half year conversations: {e}")
            raise

    async def get_last_year_conversations(
            self,
            thread_id: Optional[str] = None,
            limit: int = 100
    ) -> List[Dict[str, Any]]:
        try:
            async with self._pool_manager.get_connection() as conn:
                query = """
                   SELECT
                        ch.id,
                        ch.thread_id,
                        ch.conversation_id,
                        ch.messages,
                        ch.metadata,
                        ch.created_at,
                        ch.updated_at,
                        ch.version
                    FROM chat_history ch
                    WHERE created_at >= CURRENT_DATE - INTERVAL '365 days'  -- 365天 = 一年
                    AND created_at < CURRENT_DATE - INTERVAL '180 days'  -- 排除最近180天
                    AND thread_id = $1
                    ORDER BY ch.created_at DESC
                    LIMIT $2
                """
                records = await conn.fetch(query, thread_id, limit)
                return [
                    {
                        'id': str(record['id']),
                        'thread_id': record['thread_id'],
                        'conversation_id': record['conversation_id'],
                        'messages': record['messages'],
                        'metadata': record['metadata'],
                        'created_at': record['created_at'],
                        'updated_at': record['updated_at'],
                        'version': record['version']
                    }
                    for record in records
                ]

        except asyncpg.PostgresError as e:
            self.logger.error(f"Database error getting year conversations (exclude recent 180 days): {e}")
            raise
    async def close(self) -> None:
        """关闭资源"""
        if self._pool_manager:
            await self._pool_manager.close()
            self._initialized = False
            self.logger.info("ChatHistoryRepository closed")


class ChatManager:
    """聊天管理器（业务逻辑层）"""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self._repository: Optional[ChatHistoryRepository] = None
        self.logger = logging.getLogger(f"{__name__}.ChatManager")
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """初始化管理器"""
        async with self._lock:
            if self._repository is None:
                self._repository = ChatHistoryRepository(self.config)
                await self._repository.initialize()
                self.logger.info("ChatManager initialized")

    async def save_conversation(
            self,
            thread_id: str,
            conversation_id: str,
            messages: List[Dict[str, Any]],
            metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """保存对话（业务逻辑）"""
        if not messages:
            self.logger.warning("Attempted to save empty messages")
            return None

        await self.initialize()
        return await self._repository.save_messages(
            thread_id=thread_id,
            conversation_id=conversation_id,
            messages=messages,
            metadata=metadata
        )

    async def get_conversation(
            self,
            thread_id: str,
            conversation_id: str
    ) -> Optional[Dict[str, Any]]:
        """获取对话"""
        await self.initialize()
        return await self._repository.get_messages(thread_id, conversation_id)

    async def delete_conversation(
            self,
            thread_id: str,
            conversation_id: str
    ) -> bool:
        """删除对话"""
        await self.initialize()
        return await self._repository.delete_messages(thread_id, conversation_id)


    async def get_todays_conversations(
            self,
            thread_id: Optional[str] = None,
            limit: int = 100,
    ) -> List[Dict[str, Any]]:
        await self.initialize()
        return await self._repository.get_todays_conversations(
            thread_id=thread_id,
            limit=limit,
        )

    async def get_yesterday_conversations(
            self,
            thread_id: Optional[str] = None,
            limit: int = 100,
    ) -> List[Dict[str, Any]]:
        await self.initialize()
        return await self._repository.get_yesterday_conversations(
            thread_id=thread_id,
            limit=limit,
        )

    async def get_last_7days_conversations(
            self,
            thread_id: Optional[str] = None,
            limit: int = 100,
    ) -> List[Dict[str, Any]]:
        await self.initialize()
        return await self._repository.get_last_7days_conversations(
            thread_id=thread_id,
            limit=limit,
        )

    async def get_last_30days_conversations(
            self,
            thread_id: Optional[str] = None,
            limit: int = 100,
    ) -> List[Dict[str, Any]]:
        await self.initialize()
        return await self._repository.get_last_30days_conversations(
            thread_id=thread_id,
            limit=limit,
        )

    async def get_last_quarter_conversations(
            self,
            thread_id: Optional[str] = None,
            limit: int = 100,
    ) -> List[Dict[str, Any]]:
        await self.initialize()
        return await self._repository.get_last_quarter_conversations(
            thread_id=thread_id,
            limit=limit,
        )

    async def get_last_half_year_conversations(
            self,
            thread_id: Optional[str] = None,
            limit: int = 100,
    ) -> List[Dict[str, Any]]:
        await self.initialize()
        return await self._repository.get_last_half_year_conversations(
            thread_id=thread_id,
            limit=limit,
        )

    async def get_last_year_conversations(
            self,
            thread_id: Optional[str] = None,
            limit: int = 100,
    ) -> List[Dict[str, Any]]:
        await self.initialize()
        return await self._repository.get_last_year_conversations(
            thread_id=thread_id,
            limit=limit,
        )

    async def close(self) -> None:
        """关闭管理器"""
        async with self._lock:
            if self._repository:
                await self._repository.close()
                self._repository = None
            self.logger.info("ChatManager closed")

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# async def main():
#     import sys
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     )
#
#     async with ChatManager() as manager:
#         """
#         保存对话
#         """
#         message_id = await manager.save_conversation(
#             thread_id="user_123",
#             conversation_id="conv_456",
#             messages=[
#                 {"role": "user", "content": "Hello", "timestamp": datetime.now().isoformat()},
#                 {"role": "assistant", "content": "Hello", "timestamp": datetime.now().isoformat()},
#             ],
#             metadata={"source": "web", "language": "en"}
#         )
#         print(f"保存对话ID: {message_id}")
#
#         """
#         获取单个对话
#         """
#         conversation = await manager.get_conversation("user_123", "conv_456")
#         print(f"获取单独会话ID: {conversation}")
#
#         """
#         列出所有对话
#         """
#         conversations = await manager.get_todays_conversations(thread_id="user_123")
#         print(f" get_todays_conversations Found {len(conversations)} conversations")
#
#         # conversations = await manager.get_yesterday_conversations(thread_id="user_123")
#         # print(f"get_yesterday_conversations Found {len(conversations)} conversations")
#         #
#         # conversations = await manager.get_last_7days_conversations(thread_id="user_123")
#         # print(f"get_last_7days_conversations Found {len(conversations)} conversations")
#         #
#         # conversations = await manager.get_last_30days_conversations(thread_id="user_123")
#         # print(f"get_last_30days_conversations Found {len(conversations)} conversations")
#         #
#         # conversations = await manager.get_last_quarter_conversations(thread_id="user_123")
#         # print(f"get_last_quarter_conversations Found {len(conversations)} conversations")
#         #
#         # conversations = await manager.get_last_half_year_conversations(thread_id="user_123")
#         # print(f"get_last_half_year_conversations Found {len(conversations)} conversations")
#         #
#         # conversations = await manager.get_last_year_conversations(thread_id="user_123")
#         # print(f"get_last_year_conversations Found {len(conversations)} conversations")
#
# async def con():
#     host = "192.168.124.137"
#     port = 5432
#
#     try:
#         # 测试TCP连接
#         reader, writer = await asyncio.open_connection(host, port)
#         print(f"✓ 可以连接到 {host}:{port}")
#         writer.close()
#         await writer.wait_closed()
#         return True
#     except Exception as e:
#         print(f"✗ 无法连接到 {host}:{port}: {e}")
#         return False
# if __name__ == '__main__':
#     import asyncio
#     asyncio.run(con())
#     asyncio.run(main())






