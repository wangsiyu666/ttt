from langgraph.store.base import Item, SearchItem, NotProvided, NOT_PROVIDED, NamespacePath
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg_pool import AsyncConnectionPool
from typing import Optional, Dict, AsyncGenerator, Any, Literal
import asyncio
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class StoreFactory:
    def __init__(self):
        self._postgres_pool = None
        self._lock = asyncio.Lock()

    async def get_store(self) -> Optional[object]:
        return await self._create_postgres_store(
            config={
                "db_uri": "postgresql://postgres:123456@192.168.124.137:5432/postgres",
                "pool": {
                    "min_size": 4,
                    "max_size": 10,
                    "timeout": 30,
                    "max_idle": 300,
                    "max_lifetime": 3600,
                    "open": False
                }
            }
        )

    async def _create_postgres_store(self, config: Dict[str, Any]):
        db_uri = config.get("db_uri")

        async with self._lock:
            try:
                if self._postgres_pool is None:
                    pool = AsyncConnectionPool(
                        conninfo=db_uri,
                        **config.get("pool", {})
                    )
                    await pool.open()

                    async with pool.connection() as conn:
                        await conn.set_autocommit(True)
                        store = AsyncPostgresStore(conn)
                        await store.setup()


                    self._postgres_pool = pool
                    logger.info("Postgres saver created")
            except Exception as e:
                logger.error(f"Failed to create PostgreSQL pool: {e}")

        class PooledAsyncPostgresStore(AsyncPostgresStore):
            def __init__(self, pool):
                self._pool = pool

            @asynccontextmanager
            async def _get_connection(self) -> AsyncGenerator[AsyncPostgresStore, None]:
                async with self._pool.connection() as conn:
                    store = AsyncPostgresStore(conn)
                    store._custom_conn = conn
                    yield store

            async def aget(
                    self,
                    namespace: tuple[str, ...],
                    key: str,
                    *,
                    refresh_ttl: bool | None = None,
            ):
                async with self._get_connection() as store:
                    return await store.aget(namespace, key, refresh_ttl=refresh_ttl)

            async def asearch(
                    self,
                    namespace_prefix: tuple[str, ...],
                    /,
                    *,
                    query: str | None = None,
                    filter: dict[str, Any] | None = None,
                    limit: int = 10,
                    offset: int = 0,
                    refresh_ttl: bool | None = None,
            ):
                async with self._get_connection() as store:
                    r = await store.asearch(namespace_prefix, limit=limit)
                    return r

            async def aput(
                    self,
                    namespace: tuple[str, ...],
                    key: str,
                    value: dict[str, Any],
                    index: Literal[False] | list[str] | None = None,
                    *,
                    ttl: float | None | NotProvided = NOT_PROVIDED,
            ) -> None:
                async with self._get_connection() as store:
                    return await store.aput(namespace, key, value, index)

            async def adelete(
                    self,
                    namespace: tuple[str, ...],
                    key: str,
            ) -> None:
                async with self._get_connection() as store:
                    return await store.adelete(namespace, key)

            async def alist_namespaces(
                    self,
                    *,
                    prefix: NamespacePath | None = None,
                    suffix: NamespacePath | None = None,
                    max_depth: int | None = None,
                    limit: int = 100,
                    offset: int = 0,
            ) -> list[tuple[str, ...]]:
                async with self._get_connection() as store:
                    return await store.alist_namespaces(prefix, suffix, max_depth, limit, offset)

            async def custom_adelete(
                    self,
                    namespace: tuple[str, ...],
                    conversation_id: str,
            ) -> int:
                async with self._get_connection() as store:
                    async with store._custom_conn.cursor() as cur:
                        await cur.execute(
                            """
                            WITH deleted AS (
                                DELETE
                                FROM store
                                    WHERE prefix = %s
                                        AND key LIKE %s
                                    RETURNING *
                            )
                            SELECT COUNT(*) FROM deleted
                            """,
                            ['.'.join(namespace), conversation_id]
                            )
                        result = await cur.fetchone()
                        deleted_count = result[0] if result else 0
                    await conn.commit()
                return deleted_count

            async def custom_asearch(
                    self,
                    namespace: tuple[str, ...],
                    conversation_id: str,
            ):
                async with self._get_connection() as store:
                    async with store._custom_conn.cursor() as cur:
                        await cur.execute(
                            """
                            SELECT value FROM store
                                WHERE prefix = %s
                                AND key LIKE %s
                                order by created_at
                            """,
                            ['.'.join(namespace), conversation_id],
                        )
                        result = await cur.fetchall()
                return result


        return PooledAsyncPostgresStore(self._postgres_pool)



    async def cleanup(self):
        cleanup_tasks = []

        if self._postgres_pool:
            cleanup_tasks.append(self._postgres_pool.close())

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            logger.info("All connections cleaned up")

_store_factory = StoreFactory()

@asynccontextmanager
async def get_store() -> AsyncGenerator[object, None]:
    store = None

    try:
        store = await _store_factory.get_store()
        yield store
    finally:
        pass

async def cleanup_checkpointers():
    """Cleanup all checkpointers (call on application shutdown)."""
    await _store_factory.cleanup()











