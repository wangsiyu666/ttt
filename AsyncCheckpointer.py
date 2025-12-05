import logging
import json
import uuid
import re
import asyncio
from datetime import datetime
from email.contentmanager import get_message_content
from typing import (
    Optional,
    List,
    Tuple,
    Dict,
    Any,
    cast
)
import asyncpg
from contextlib import asynccontextmanager
from langgraph.store.memory import InMemoryStore
from sqlalchemy.sql.functions import current_timestamp
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage, AIMessageChunk, BaseMessage

from milvus_retriever.retriever import Resource

logger = logging.getLogger(__name__)
class AsyncChatStreamManager:
    def __init__(
            self,
            checkpointer_saver: bool = False,
            db_uri: Optional[str] = None
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.checkpointer_saver = checkpointer_saver
        self.db_uri = db_uri
        self.store = InMemoryStore()
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self) -> None:
        if self.checkpointer_saver:
            await self._init_postgresql()
        else:
            self.logger.warning(
                "Checkpoint saver is disabled"
            )

    async def _init_postgresql(self) -> None:
        try:
            self.postgres_pool = await asyncpg.create_pool(
                self.db_uri
            )
            self.logger.info("Connected to PostgreSQL database")
            await self._create_chat_streams_table()
        except Exception as e:
            self.logger.error(f"Error connecting to PostgreSQL database: {str(e)}")
            raise

    async def _create_chat_streams_table(self) -> None:
        if not self.postgres_pool:
            raise RuntimeError("Database pool not initialized")

        try:
            async with self.postgres_pool.acquire() as connection:
                async with connection.transaction():
                    create_tables_sql = """
                    CREATE TABLE IF NOT EXISTS chat_streams (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        thread_id VARCHAR(255) NOT NULL UNIQUE,
                        messages JSONB NOT NULL,
                        ts TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_chat_streams_thread_id ON chat_streams (thread_id);
                    CREATE INDEX IF NOT EXISTS idx_chat_streams_ts ON chat_streams (ts);
                    """
                    await connection.execute(create_tables_sql)
                    self.logger.info(
                        "Chat streams table created/verified successfully"
                    )
        except Exception as e:
            self.logger.error(
                f"Failed to create chat streams table: {str(e)}"
            )
            raise
    async def process_stream_message(
            self,
            thread_id: str,
            message: str,
            finish_reason: str
    ) -> bool:
        if not thread_id or not isinstance(thread_id, str):
            self.logger.warning("Invalid thread_id provided")
            return False
        if not message:
            self.logger.warning("Invalid message provided")
            return False

        try:
            store_namespace: Tuple[str, str] = ("messages", thread_id)

            cursor = self.store.get(store_namespace, "cursor")
            current_index = 0

            if cursor is None:
                self.store.put(store_namespace, "cursor", {"index": 0})
            else:
                current_index = int(cursor.value.get("index", 0)) + 1
                self.store.put(store_namespace, "cursor", {"index": current_index})

            self.store.put(store_namespace, f"chunk_{current_index}", message)

            if finish_reason in ("stop", "interrupt"):
                return await self._persist_complete_conversation(
                    thread_id,
                    store_namespace,
                    current_index
                )
            return True
        except Exception as e:
            self.logger.error(
                f"Error processing stream message for thread {thread_id}: {e}"
            )
            return False
    async def _persist_complete_conversation(
            self,
            thread_id: str,
            store_namespace: Tuple[str, str],
            final_index: int
    ) -> bool:
        memories = self.store.search(
            store_namespace,
            limit=final_index + 2
        )

        try:
            messages: List[str] = []
            for item in memories:
                value = item.dict().get("value", "")

                if value and not isinstance(value, dict):
                    messages.append(str(value))

            if not messages:
                self.logger.warning(
                    "Checkpointer saver is disabled"
                )
                return False

            if self.postgres_pool is not None:
                return await self._persist_to_postgresql(
                    thread_id,
                    messages
                )
            else:
                self.logger.warning(
                    "No database connection available for Memory"
                )
                return False
        except Exception as e:
            self.logger.error(
                f"Error persisting conversation for thread {thread_id}: {str(e)}"
            )
            return False

    async def _persist_to_postgresql(
            self,
            thread_id: str,
            messages: List[str]
    ) -> bool:
        if not self.postgres_pool:
            self.logger.error(
                "Database pool not initialized"
            )
            return False

        try:
            async with self.postgres_pool.acquire() as connection:
                async with connection.transaction():
                    existing_record = await connection.fetchrow(
                        "SELECT id FROM chat_streams WHERE thread_id = $1",
                        thread_id
                    )

                    current_timestamp = datetime.now()
                    messages_json = json.dumps(messages)

                    if existing_record is None:
                        result = await connection.execute(
                            """
                            UPDATE chat_streams
                                SET messages = $1, ts = $2
                                WHERE thread_id = $3
                            """,
                            messages_json, current_timestamp, thread_id
                        )

                        affected_rows = int(result.split()[-1])
                        self.logger.info(
                            f"Updated conversation for thread {thread_id}:"
                            f"{affected_rows} rows modified"
                        )
                        return affected_rows > 0
                    else:
                        conversation_id = uuid.uuid4()
                        result = await connection.execute(
                            """
                            INSERT INTO chat_streams (id, thread_id, messages, ts)
                                VALUES ($1, $2, $3, $4)
                            """,
                            conversation_id, thread_id, messages_json, current_timestamp
                        )

                        affected_rows = int(result.split()[-1])

                        self.logger.info(
                            f"Created new conversation with ID: {conversation_id}"
                        )
                        return affected_rows > 0
        except Exception as e:
            self.logger.error(
                f"Error persisting to PostgreSQL: {str(e)}"
            )
            return False

    async def get_conversation(self, thread_id: str) -> Optional[List[str]]:
        """异步获取对话历史"""
        if not self.postgres_pool:
            self.logger.error("Database pool not initialized")
            return None

        try:
            async with self.postgres_pool.acquire() as connection:
                record = await connection.fetchrow(
                    "SELECT messages FROM chat_streams WHERE thread_id = $1",
                    thread_id
                )

                if record:
                    return json.loads(record['messages'])
                return None
        except Exception as e:
            self.logger.error(f"Error getting conversation for thread {thread_id}: {e}")
            return None

    async def close(self) -> None:
        try:
            if self.postgres_pool:
                await self.postgres_pool.close()
                self.logger.info("PostgreSQL connection pool closed")
        except Exception as e:
            self.logger.error(f"Error closing PostgreSQL: {e}")

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


LANGGRAPH_CHECKPOINT_SAVER = ""
LANGGRAPH_CHECKPOINT_URI = "postgres://postgres:postgres@192.168.124.137:5432/postgres"


_default_manager: Optional[AsyncChatStreamManager] = None


async def chat_stream_message(thread_id: str, message: str, finish_reason: str) -> bool:
    """异步处理聊天流消息"""
    checkpointer_saver = LANGGRAPH_CHECKPOINT_SAVER
    if checkpointer_saver:
        manager = await chat_stream_message()
        return await manager.process_stream_message(
            thread_id,
            message,
            finish_reason
        )
    else:
        return False

async def get_chat_history(thread_id: str) -> Optional[List[str]]:
    """异步获取聊天历史"""
    checkpointer_saver = LANGGRAPH_CHECKPOINT_SAVER
    if checkpointer_saver:
        manager = await chat_stream_message()
        return await manager.get_conversation(thread_id)
    return None


async def _process_initial_messages(message, thread_id):
    json_data = json.dumps(
        {
            "thread_id": thread_id,
            "id": "run--" + message.get("id", uuid.uuid4().hex),
            "role": "user",
            "content": message.get("content", ""),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    await chat_stream_message(
        thread_id,
        f"event: message_chunk\ndata: {json_data}\n\n",
        "none"
    )


async def _make_event(event_type: str, data: dict[str, any]):
    if data.get("content") == "":
        data.pop("content")

    try:
        json_data = json.dumps(data, ensure_ascii=False)

        finish_reason = data.get("finish_reason", "")
        await chat_stream_message(
            data.get("thread_id", ""),
            f"event: {event_type}\ndata: {json_data}\n\n",
            finish_reason
        )
        return f"event: {event_type}\ndata: {json_data}\n\n"

    except (TypeError, ValueError) as e:
        logger.error(
            f"Error serializing event data: {str(e)}"
        )
        error_data = json.dumps(
            {
                "error": "Serialization failed"
            },
            ensure_ascii=False,
        )
        return f"event: error\ndata: {error_data}\n\n"


def get_ChatOpenAI():
    from langchain_openai.chat_models.base import BaseChatOpenAI
    model = BaseChatOpenAI(
        model_name="Qwen/Qwen3-VL-235B-A22B-Instruct",
        base_url="https://api.siliconflow.cn/v1",
        api_key="sk-jeybgrflkhxmtscojwsukplimnpyojrvxhwdgrzmetuhemev"
    )
    return model

def _create_event_stream_message(
        message_chunk,
        message_metadata,
        thread_id,
        agent_name
):
    content = message_chunk.content
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    event_stream_message = {
        "thread_id": thread_id,
        "agent": agent_name,
        "id": message_chunk.id,
        "role": "assistant",
        "checkpoint_ns": message_metadata.get("checkpoint_ns", ''),
        "langgraph_node": message_metadata.get("langgraph_node", ''),
        "langgraph_path": message_metadata.get("langgraph_path", ''),
        "langgraph_step": message_metadata.get("langgraph_step", ''),
        "content": content,
    }
    if message_chunk.additional_kwargs.get("reasoning_content"):
        event_stream_message["reasoning_content"] = message_chunk.additional_kwargs["reasoning_content"]

    if message_chunk.additional_kwargs.get("finish_reason"):
        event_stream_message["finish_reason"] = message_chunk.response_metadata.get("finish_reason")

    return event_stream_message

def _get_agent_name(agent, message_metadata):
    agent_name = "unknown"
    if agent and len(agent) > 0:
        agent_name = agent[0].split(":")[0] if ":" in agent[0] else agent[0]
    else:
        agent_name = message_metadata.get("langgraph_node", "unknown")
    return agent_name

def sanitize_log_input(value: Any, max_length: int = 500) -> str:
    """
        Sanitize user-controlled input for safe logging.

    Replaces dangerous characters (newlines, tabs, carriage returns, etc.)
    with their escaped representations to prevent log injection attacks.

    Args:
        value: The input value to sanitize (any type)
        max_length: Maximum length of output string (truncates if exceeded)

    Returns:
        str: Sanitized string safe for logging

    Examples:
        >>> sanitize_log_input("normal text")
        'normal text'

        >>> sanitize_log_input("malicious\n[INFO] fake entry")
        'malicious\\n[INFO] fake entry'

        >>> sanitize_log_input("tab\there")
        'tab\\there'

        >>> sanitize_log_input(None)
        'None'

        >>> long_text = "a" * 1000
        >>> result = sanitize_log_input(long_text, max_length=100)
        >>> len(result) <= 100
        True
    """
    if value is None:
        return None

    string_value = str(value)

    replacements = {
        "\\": "\\\\",
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
        "\x00": "\\0",
        "\x1b": "\\x1b",
    }
    for char, replacement in replacements.items():
        string_value = string_value.replace(char, replacement)
    string_value = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", "", string_value)

    if len(string_value) > max_length:
        string_value = string_value[: max_length - 3] + "..."

    return string_value

def sanitize_thread_id(thread_id: Any) -> str:
    return sanitize_log_input(thread_id, max_length=100)

def sanitize_user_content(content: Any) -> str:
    return sanitize_log_input(content, max_length=200)

def sanitize_agent_name(agent_name: str) -> str:
    return sanitize_log_input(agent_name, max_length=100)

def sanitize_tool_name(tool_name: str) -> str:
    return sanitize_log_input(tool_name, max_length=100)

def sanitize_args(args: Any) -> str:
    if not isinstance(args, str):
        return ""
    else:
        return (
            args.replace("[", "&#91;")
            .replace("]", "&#93;")
            .replace("{", "&#123;")
            .replace("}", "&#125;")
        )

def create_safe_log_message(template: str, **kwargs) -> str:
    """
    Create a safe log message by sanitizing all values.

    Uses a template string with keyword arguments, sanitizing each value
    before substitution to prevent log injection.

    Args:
        template: Template string with {key} placeholders
        **kwargs: Key-value pairs to substitute

    Returns:
        str: Safe log message

    Example:
        >>> msg = create_safe_log_message(
        ...     "[{thread_id}] Processing {tool_name}",
        ...     thread_id="abc\\n[INFO]",
        ...     tool_name="my_tool"
        ... )
        >>> "[abc\\\\n[INFO]] Processing my_tool" in msg
        True
    """
    safe_kwargs = {
        key: sanitize_log_input(value) for key, value in kwargs.items()
    }
    return template.format(**safe_kwargs)

def is_user_message(message: Any) -> bool:
    ASSISTANT_SPEAKER_NAMES = {
        "coordinator",
        "planner",
        "researcher",
        "coder",
        "reporter",
        "background_investigator",
    }
    if isinstance(message, dict):
        role = (message.get("role") or "").lower()
        if role in {"user", "human"}:
            return True
        if role in {"assistant", "system"}:
            return False
        name = (message.get("name") or "").lower()
        if name and name in ASSISTANT_SPEAKER_NAMES:
            return False
        return role == "" and name not in ASSISTANT_SPEAKER_NAMES
    message_type = (getattr(message, "type", "") or "").lower()
    name = (getattr(message, "name", "") or "").lower()
    if message_type == "human":
        return not (name and name in ASSISTANT_SPEAKER_NAMES)

    role_attr = getattr(message, "role", None)
    if isinstance(role_attr, str) and role_attr.lower() in {"user", "human"}:
        return True

    additional_role = getattr(message, "additional_kwargs", {}).get("role")
    if isinstance(additional_role, str) and additional_role.lower() in {
        "user",
        "human",
    }:
        return True

    return False


def reconstruct_clarification_history(
        messages: list[Any],
        fallback_history: list[str] | None = None,
        base_topic: str = ""
) -> list[str]:
    sequence: list[str] = []
    for message in messages or []:
        if not is_user_message(message):
            continue
        content = get_message_content(message)
        if not content:
            continue

        if sequence and sequence[-1] == content:
            continue
        sequence.append(content)

    if sequence:
        return sequence

    fallback = [item for item in (fallback_history or []) if item]
    if fallback:
        return fallback

    base_topic = (base_topic or "").strip()
    return [base_topic] if base_topic else []

def _validate_tool_call_chunks(tool_call_chunks):
    if not tool_call_chunks:
        return

    indices_seen = set()
    tool_ids_seen = set()

    for i, chunk in enumerate(tool_call_chunks):
        index = chunk.get("index")
        tool_id = chunk.get("id")
        name = chunk.get("name", "")
        has_args = "args" in chunk

        logger.debug(
            f"Chunk {i}: index={index}, id={tool_id}, name={name}, "
            f"has_args={has_args}, type={chunk.get('type')}"
        )
        if index is not None:
            indices_seen.add(index)
        if tool_id:
            tool_ids_seen.add(tool_id)

    if len(indices_seen) > 1:
        logger.debug(
            f"Multiple indices detected: {sorted(indices_seen)} - "
            f"This may indicate consecutive tool calls"
        )

def _process_tool_call_chunks(tool_call_chunks):
    if not tool_call_chunks:
        return []
    _validate_tool_call_chunks(tool_call_chunks)

    chunks = []
    chunk_by_index = {}

    for chunk in tool_call_chunks:
        index = chunk.get("index")
        chunk_id = chunk.get("id")

        if index is not None:
            if index not in chunk_by_index:
                chunk_by_index[index] = {
                    "name": "",
                    "args": "",
                    "id": chunk_id or "",
                    "index": index,
                    "type": chunk.get("type", ""),
                }
            chunk_name = chunk.get("name", "")
            if chunk_name:
                stored_name = chunk_by_index[index]["name"]

                if stored_name and stored_name != chunk_name:
                    logger.warning(
                        f"Tool name mismatch detected at index {index}: "
                        f"'{stored_name}' != '{chunk_name}'. "
                        f"This may indicate a streaming artifact or consecutive tool calls "
                        f"with the same index assignment."
                    )
                else:
                    chunk_by_index[index]["name"] = chunk_name
            if chunk_id and not chunk_by_index[index]["name"]:
                chunk_by_index[index]["name"] = chunk_id

            if chunk.get("args"):
                chunk_by_index[index]["args"] += chunk.get("args", "")
        else:
            logger.debug(f"Chunk without index encountered: {chunk}")
            chunks.append(
                {
                    "name": chunk.get("name", ""),
                    "args": sanitize_args(chunk.get("args", "")),
                    "id": chunk.get("id", ""),
                    "index": 0,
                    "type": chunk.get("type", ""),
                }
            )
    for index in sorted(chunk_by_index.keys()):
        chunk_data = chunk_by_index[index]
        chunk_data["args"] = sanitize_args(chunk_data["args"])
        chunks.append(chunk_data)
        logger.debug(
            f"Processed tool call: index={index}, name={chunk_data['name']}, "
            f"id={chunk_data['id']}"
        )
    return chunks

async def _process_message_chunk(message_chunk, message_metadata, thread_id, agent):
    """Process a single message chunk and yield appropriate events."""

    agent_name = _get_agent_name(agent, message_metadata)
    safe_agent_name = sanitize_agent_name(agent_name)
    safe_thread_id = sanitize_thread_id(thread_id)
    safe_agent = sanitize_agent_name(agent)
    logger.debug(f"[{safe_thread_id}] _process_message_chunk started for agent={safe_agent_name}")
    logger.debug(f"[{safe_thread_id}] Extracted agent_name: {safe_agent_name}")

    event_stream_message = _create_event_stream_message(
        message_chunk, message_metadata, thread_id, agent_name
    )

    if isinstance(message_chunk, ToolMessage):
        # Tool Message - Return the result of the tool call
        logger.debug(f"[{safe_thread_id}] Processing ToolMessage")
        tool_call_id = message_chunk.tool_call_id
        event_stream_message["tool_call_id"] = tool_call_id

        # Validate tool_call_id for debugging
        if tool_call_id:
            safe_tool_id = sanitize_log_input(tool_call_id, max_length=100)
            logger.debug(f"[{safe_thread_id}] ToolMessage with tool_call_id: {safe_tool_id}")
        else:
            logger.warning(f"[{safe_thread_id}] ToolMessage received without tool_call_id")

        logger.debug(f"[{safe_thread_id}] Yielding tool_call_result event")
        yield _make_event("tool_call_result", event_stream_message)
    elif isinstance(message_chunk, AIMessageChunk):
        # AI Message - Raw message tokens
        has_tool_calls = bool(message_chunk.tool_calls)
        has_chunks = bool(message_chunk.tool_call_chunks)
        logger.debug(
            f"[{safe_thread_id}] Processing AIMessageChunk, tool_calls={has_tool_calls}, tool_call_chunks={has_chunks}")

        if message_chunk.tool_calls:
            # AI Message - Tool Call (complete tool calls)
            safe_tool_names = [sanitize_tool_name(tc.get('name', 'unknown')) for tc in message_chunk.tool_calls]
            logger.debug(f"[{safe_thread_id}] AIMessageChunk has complete tool_calls: {safe_tool_names}")
            event_stream_message["tool_calls"] = message_chunk.tool_calls

            # Process tool_call_chunks with proper index-based grouping
            processed_chunks = _process_tool_call_chunks(
                message_chunk.tool_call_chunks
            )
            if processed_chunks:
                event_stream_message["tool_call_chunks"] = processed_chunks
                safe_chunk_names = [sanitize_tool_name(c.get('name')) for c in processed_chunks]
                logger.debug(
                    f"[{safe_thread_id}] Tool calls: {safe_tool_names}, "
                    f"Processed chunks: {len(processed_chunks)}"
                )

            logger.debug(f"[{safe_thread_id}] Yielding tool_calls event")
            yield _make_event("tool_calls", event_stream_message)
        elif message_chunk.tool_call_chunks:
            # AI Message - Tool Call Chunks (streaming)
            chunks_count = len(message_chunk.tool_call_chunks)
            logger.debug(f"[{safe_thread_id}] AIMessageChunk has streaming tool_call_chunks: {chunks_count} chunks")
            processed_chunks = _process_tool_call_chunks(
                message_chunk.tool_call_chunks
            )

            # Emit separate events for chunks with different indices (tool call boundaries)
            if processed_chunks:
                prev_chunk = None
                for chunk in processed_chunks:
                    current_index = chunk.get("index")

                    # Log index transitions to detect tool call boundaries
                    if prev_chunk is not None and current_index != prev_chunk.get("index"):
                        prev_name = sanitize_tool_name(prev_chunk.get('name'))
                        curr_name = sanitize_tool_name(chunk.get('name'))
                        logger.debug(
                            f"[{safe_thread_id}] Tool call boundary detected: "
                            f"index {prev_chunk.get('index')} ({prev_name}) -> "
                            f"{current_index} ({curr_name})"
                        )

                    prev_chunk = chunk

                # Include all processed chunks in the event
                event_stream_message["tool_call_chunks"] = processed_chunks
                safe_chunk_names = [sanitize_tool_name(c.get('name')) for c in processed_chunks]
                logger.debug(
                    f"[{safe_thread_id}] Streamed {len(processed_chunks)} tool call chunk(s): "
                    f"{safe_chunk_names}"
                )

            logger.debug(f"[{safe_thread_id}] Yielding tool_call_chunks event")
            yield _make_event("tool_call_chunks", event_stream_message)
        else:
            # AI Message - Raw message tokens
            content_len = len(message_chunk.content) if isinstance(message_chunk.content, str) else 0
            logger.debug(f"[{safe_thread_id}] AIMessageChunk is raw message tokens, content_len={content_len}")
            yield _make_event("message_chunk", event_stream_message)

async def _stream_graph_events(
        graph_instance,
        workflow_input,
        workflow_config,
        thread_id
):
    safe_thread_id = sanitize_thread_id(thread_id)
    logger.debug(f"[{safe_thread_id}] Starting graph event stream with agent nodes")
    try:
        event_count = 0
        async for agent, _, event_data in graph_instance.astream(
            workflow_input,
            config=workflow_config,
            stream_mode=["messages", "updates"],
            subgraph=True,
        ):
            event_count += 1
            safe_agent = sanitize_agent_name(agent)
            logger.debug(f"[{safe_thread_id}] Graph event #{event_count} received from agent: {safe_agent}")

            if isinstance(event_data, dict):
                if "__interrupt__" in event_data:
                    ...
                continue
            message_chunk, message_metadata = cast(
                tuple[BaseMessage, dict[str, Any]],
                event_data
            )
            safe_node = sanitize_agent_name(message_metadata.get("langgraph_node", "unknown"))
            safe_step = sanitize_log_input(message_metadata.get("langgraph_step", "unknown"))
            logger.debug(
                f"[{safe_thread_id}] Processing message chunk: "
                f"type={type(message_chunk).__name__}, "
                f"node={safe_node}, "
                f"step={safe_step}"
            )
            async for event in _process_message_chunk(
                message_chunk, message_metadata, thread_id, agent
            ):
                yield event
        logger.debug(f"[{safe_thread_id}] Graph event stream completed. Total events: {event_count}")
    except asyncio.CancelledError:
        # User cancelled/interrupted the stream - this is normal, not an error
        logger.info(f"[{safe_thread_id}] Graph event stream cancelled by user after {event_count} events")
        # Re-raise to signal cancellation properly without yielding an error event
        raise

    except Exception as e:
        logger.exception(f"[{safe_thread_id}] Error during graph execution")
        yield _make_event(
    "error",
    {
        "thread_id": thread_id,
        "error": "Error during graph execution",
        },
    )

async def _astream_workflow_generator(
        messages: List[dict],
        thread_id: str,
        resources: List[Resource],
        max_plan_iterations: int,
        max_step_num: int,
        max_search_results: int,
        auto_accepted_plan: bool,
        interrupt_feedback: str,
        mcp_setting: dict,
        enable_background_investigation: bool,
        report_style: ReportStyle,
        enable_deep_thinking: bool,
        enable_clarification_rounds: int,
        locale: str = "en-US",
        interrupt_before_tools: Optional[List[str]] = None
):
    safe_thread_id = sanitize_thread_id(thread_id)
    safe_feedback = sanitize_log_input(interrupt_feedback) if interrupt_feedback else ""

    logger.debug(
        f"[{safe_thread_id}] _astream_workflow_generator starting: "
        f"messages_count={len(messages)}, "
        f"auto_accepted_plan={auto_accepted_plan}, "
        f"interrupt_feedback={safe_feedback}, "
        f"interrupt_before_tools={interrupt_before_tools}"
    )
    logger.debug(f"[{safe_thread_id}] Processing {len(messages)} initial messages")

    for message in messages:
        if isinstance(message, dict) and "content" in messages:
            safe_content = sanitize_user_content(message.get("content", ''))
            logger.debug(f"[{safe_thread_id}] Sending initial message to client: {safe_content}")
            _process_initial_messages(message, thread_id)
    logger.debug(f"[{safe_thread_id}] Reconstructing clarification history")
    clarification_history = reconstruct_clarification_history(messages)

    logger.debug(f"[{safe_thread_id}] Building clarified topic from history")
















