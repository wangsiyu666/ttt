"""
agent执行历史
"""
import logging
from typing import Optional
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from dotenv import load_dotenv
import os

from tests.test_init import kwargs

load_dotenv()

logger = logging.getLogger(__name__)
_connection_pool = {}

async def _create_checkpointer_from_config() -> Optional[object]:
    global _connection_pool
    db_uri = os.getenv("DB_URI")
    config = {
        "BASIC_MODEL": {
            "base_url": "",
            "model": "",
            "api_key": ""
        },
        "STORAGE": {
            "type": "postgres",
            "db_uri": db_uri
        }
    }

    if "postgres" not in _connection_pool:
        try:
            conn_params = {}
            for param in db_uri.split():
                if "=" in param:
                    key, value = param.split("=", 1)
                    conn_params[key] = value

            min_size = int(os.getenv("POOL_MIN_SIZE"))
            max_size = int(os.getenv("POOL_MAX_SIZE"))
            timeout = int(os.getenv("POOL_TIMEOUT"))
            max_idle = int(os.getenv("POOL_MAX_IDLE"))
            max_lifetime = int(os.getenv("POOL_MAX_LIFETIME"))
            pool_name = os.getenv("POOL_NAME")
            """
            注意需要修改postgres配制文件以允许外部连接
            """
            pool = AsyncConnectionPool(
                kwargs=conn_params,
                min_size=min_size,
                max_size=max_size,
                timeout=timeout,
                max_idle=max_idle,
                max_lifetime=max_lifetime,
                name=pool_name,
                open=False
            )

            await pool.open()

            async def check_table_exists():
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        check_sql = """
                        SELECT 
                            (EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'checkpoints')) AND
                            (EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'checkpoint_migrations')) AND
                            (EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'checkpoint_writes')) AND
                            (EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'checkpoint_blobs'))
                            AS all_tables_exist;
                        """
                        await cur.execute(check_sql)
                        result = await cur.fetchone()
                        return result

            table_exists = await check_table_exists()
            if not table_exists[0]:
                async with pool.connection() as conn:
                    await conn.set_autocommit(True)
                    saver = AsyncPostgresSaver(conn=conn)
                    await saver.setup()
                    logger.info("Database tables created successfully")
            else:
                logger.info("Database tables already exist, skipping setup()")

            class PooledPostgresSaver(AsyncPostgresSaver):
                def __init__(self, pool):
                    self.pool = pool

                async def aget_tuple(self, config):
                    async with self.pool.connection() as conn:
                        temp_saver = AsyncPostgresSaver(conn=conn)
                        return await temp_saver.aget_tuple(config)

                async def aput(self, config, checkpointer, metadata, new_versions):
                    async with self.pool.connection() as conn:
                        temp_saver = AsyncPostgresSaver(conn=conn)
                        return await temp_saver.aput(
                            config,
                            checkpointer,
                            metadata,
                            new_versions
                        )
                async def alist(self, config, **kwargs):
                    async with self.pool.connection() as conn:
                        temp_saver = AsyncPostgresSaver(conn=conn)
                        async for item in temp_saver.alist(config, **kwargs):
                            yield item

                async def aput_writes(self, config, writes, task_id, task_path=""):
                    async with self.pool.connection() as conn:
                        temp_saver = AsyncPostgresSaver(conn=conn)
                        await temp_saver.aput_writes(config, writes, task_id, task_path)

                async def adelete_thread(self, thread_id):
                    async with self.pool.connection() as conn:
                        temp_saver = AsyncPostgresSaver(conn=conn)
                        await temp_saver.adelete_thread(thread_id)
            pooled_saver = PooledPostgresSaver(pool)

            _connection_pool["postgresql"] = {
                "pool": pool,
                "saver": pooled_saver
            }
            return pooled_saver
        except Exception as e:
            logger.error(f"Error setting up Postgresql connection: {str(e)}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
    else:
        logger.info("Reusing existing PostgreSQL connection pool")
        return _connection_pool["postgresql"]["saver"]


async def main():
    memory = await _create_checkpointer_from_config()
    return memory

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())











