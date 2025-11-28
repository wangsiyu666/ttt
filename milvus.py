import hashlib
import logging
import os
import uuid
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Iterable,
    Optional,
    Sequence,
    Set
)

import numpy as np
import requests
from langchain_milvus.vectorstores import Milvus as LangchainMilvus
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

from dotenv import load_dotenv
import logging

from retriever import Chunk, Document, Resource, Retriever

load_dotenv()
logger = logging.getLogger(__name__)


class EmbeddingBase:
    def __init__(
            self,
            api_url: str,
            api_key: Optional[str] = None,
            model_name: Optional[str] = None,
            timeout: int = 30,
            extra_headers: Optional[Dict[str, str]] = None,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.extra_headers = extra_headers or {}
        self.embedding_dim = None

    def encode_documents(self, documents: List[str]) -> np.ndarray:
        if not documents:
            return np.zeros((0, self.embedding_dim or 768), dtype=np.float32)
        return self._embed(documents)

    def encode_queries(self, queries: List[List[str]]) -> np.ndarray:
        if not queries:
            return np.zeros((0, self.embedding_dim or 768), dtype=np.float32)
        return self._embed(queries)

    def _embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError("子类必须实现 _embed()")

    def _request_json(self, method: str, url: str, **kwargs) -> Any:
        headers = dict(self.extra_headers)
        if self.api_key:
            headers.setdefault("Authorization", f"Bearer {self.api_key}")
        headers.setdefault("Content-Type", "application/json")

        resp = requests.request(
            method,
            url,
            headers=headers,
            timeout=self.timeout,
            **kwargs
        )

        try:
            resp.raise_for_status()
        except  requests.exceptions.HTTPError as e:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}") from e

        try:
            return resp.json()
        except Exception as e:
            raise RuntimeError(f"Invalid JSON response: {resp.text[:400]}") from e


class JinaEmbedding(EmbeddingBase):
    def __init__(
            self,
            api_url: str = "http://127.0.0.1:8603/v1/embeddings",
            api_key: Optional[str] = None,
            model_name: Optional[str] = None,
            timeout: int = 30,
            normalize: bool = True,
            pooling: str = "mean",
            extra_headers: Optional[Dict[str, str]] = None,
            skip_test: bool = False,
    ):
        super().__init__(
            api_url,
            api_key,
            model_name,
            timeout,
            extra_headers
        )
        self.normalize = normalize
        self.pooling = pooling

        if not skip_test:
            try:
                test_emb = self._embed(["ping"])
                self.embedding_dim = test_emb.shape[1]
                logger.info(f"Jina 连接成功: {api_url} (维度: {self.embedding_dim})")
            except Exception as e:
                logger.warning(f"Jina 连接测试失败")

    def _embed(self, texts: List[str]) -> np.ndarray:
        payload = {
            "inputs": [{"text": t} for t in texts],
            "normalize": self.normalize,
            "pooling": self.pooling,
        }

        if self.model_name:
            payload["model"] = self.model_name

        data = self._request_json("POST", self.api_url, json=payload)
        embs = data.get("embeddings")
        if not embs:
            raise RuntimeError(f"Empty embeddings from jsin: {data}")

        result = np.array(embs, dtype=np.float32)

        if self.embedding_dim is None:
            self.embedding_dim = result.shape[1]
        return result


class QwenEmbedding(EmbeddingBase):
    def __init__(
            self,
            api_url: str = "http://192.168.49.81:31893/v1/embeddings",
            api_key: Optional[str] = None,
            model_name: Optional[str] = "Qwen3-Embedding-0.6B",
            timeout: int = 30,
            extra_headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            api_url,
            api_key,
            model_name,
            timeout,
            extra_headers
        )

    def _embed(self, texts: List[str]) -> np.ndarray:
        url = f"{self.api_url}"
        payload = {"input": texts}

        if self.model_name:
            payload["model"] = self.model_name

        data = self._request_json("POST", url, json=payload)
        items = data.get("data")
        if not items:
            raise RuntimeError(f"Empty embeddings from qwen: {data}")

        embs = [it.get("embedding") for it in items]
        result = np.array(embs, dtype=np.float32)

        if self.embedding_dim is None:
            self.embedding_dim = result.shape[1]

        return result

# DEFAULT_MILVUS_URI = "./milvus.db"
DEFAULT_MILVUS_URI = "http://192.168.124.137:19530"
DEFAULT_USER = ""
DEFAULT_PWD = ""
DEFAULT_COLLECTION = "test"
"""
L2: 欧几里得
IP: 点积
COSINE: 余弦
"""
METRIC_TYPE = "COSINE"
TOP_N = 10

class MilvusRetriever(Retriever):
    def __init__(self):
        self.metric_type = METRIC_TYPE
        self.embedding_function = QwenEmbedding()
        self._embedding_dim = self.embedding_function.encode_documents(["wsy"])[0].shape[0]
        self.n_results = TOP_N
        self._create_collections()
        connection_args = {
            "uri": DEFAULT_MILVUS_URI,
            "user": DEFAULT_USER,
            "password": DEFAULT_PWD
        }
        self.milvus_client = LangchainMilvus(
            embedding_function=self.embedding_function,
            collection_name=DEFAULT_COLLECTION,
            connection_args=connection_args,
            drop_old=False
        )

    def _create_collections(self):
        self._create_default_collections("default")
        self._create_test_collections("test")
        # todo custom collections

    def generate_embedding(self, data: str, **kwargs) -> List[float]:
        return self.embedding_function.encode_documents(data).tolist()

    def _create_default_collections(self, name: str):
        self._connect()
        if not self.milvus_client.has_collection(collection_name=name):
            default_schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=False
            )
            default_schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                max_length=65535,
                is_primary=True
            )
            default_schema.add_field(
                field_name="text",
                datatype=DataType.VARCHAR,
                max_length=65535
            )
            default_schema.add_field(
                field_name="documents",
                datatype=DataType.VARCHAR,
                max_length=65535,
            )
            default_schema.add_field(
                field_name="vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=self._embedding_dim,
            )
            default_index_params = self.milvus_client.prepare_index_params()
            default_index_params.add_index(
                field_name="vector",
                index_name="vector",
                index_params="AUTOINDEX",
                metric_type=self.metric_type,
            )

            self.milvus_client.create_collection(
                collection_name=name,
                schema=default_schema,
                index_params=default_index_params,
                consistency_level="Strong"
            )
            logger.info(f"Created collection: {name} (metric_type: {self.metric_type})")

    def _create_test_collections(self, name: str):
        if not self.milvus_client.has_collection(collection_name=name):
            test_schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=False
            )

            test_schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                max_length=512,
                is_primary=True
            )
            test_schema.add_field(
                field_name="embedding",
                datatype=DataType.FLOAT_VECTOR,
                dim=self._embedding_dim,
            )
            test_schema.add_field(
                field_name="content",
                datatype=DataType.VARCHAR,
                max_length=65535,
            )
            test_schema.add_field(
                field_name="title",
                datatype=DataType.VARCHAR,
                max_length=512
            )
            test_schema.add_field(
                field_name="url",
                datatype=DataType.FLOAT_VECTOR,
            )
            test_index_params = self.milvus_client.prepare_index_params()
            test_index_params.add_index(
                field_name="id",
                index_name="id",
                index_type="AUTOINDEX",
                metric_type=self.metric_type,
            )

            self.milvus_client.create_collection(
                collection_name=name,
                schema=test_schema,
                index_params=test_index_params,
                consistency_level="Strong"
            )
            logger.info(f"Created collection: {name} (metric_type: {self.metric_type})")

    # todo _create_customs_collections

    def add_documentation(self, documentation: str, **kwargs) -> str:
        if len(documentation) == 0:
            raise Exception("documentation can not be empty")

        _id = str(uuid.uuid4()) + "-doc"
        embedding = self.embedding_function.encode_documents([documentation])[0]
        self.milvus_client.insert(
            collection_name="default",
            data={
                "id": _id,
                "doc": documentation,
                "vector": embedding
            }
        )
        return _id

    def add_test(
            self,
            doc_id: str,
            content: str,
            title: str,
            url: str,
            metadata: Dict[str, Any],
    ) -> None:
        try:
            self.milvus_client.add_texts(
                texts=[content],
                metadatas=[
                    {
                        "id": doc_id,
                        "title": title,
                        "url": url,
                        **metadata
                    }
                ],
            )
        except Exception as e:
            raise RuntimeError(f"Failed to add test: {e}")

    # todo add_custom

    def get_similar_question(self, question: str, **kwargs) -> list:
        search_params = {
            "metric_type": self.metric_type,
            "params": {"nprobe": 128}
        }
        embeddings = self.embedding_function.encode_queries([question])
        res = self.milvus_client.search(
            collection_name="default",
            anns_field="vector",
            data=embeddings,
            limit=self.n_results,
            output_fields=["text", "document"],
            search_params=search_params
        )
        res = res[0]

        list_documents = []
        for doc in res:
            dict = {}
            dict["question"] = doc["entity"]["text"]
            dict["document"] = doc["entity"]["document"]
            list_documents.append(dict)
        return list_documents

    def remove_data(self, id: str, **kwargs) -> bool:
        if id.endswith("-doc"):
            self.milvus_client.delete(collection_name="default", ids=[id])
            return True
        # todo custom endswith
        else:
            return False

    def list_resources(self, query: Optional[str] = None) -> List[Resource]:

        try:
            if self._is_milvus_lite():
                resources: List[Resource] = []
                result = self.milvus_client.query(
                    collection_name="test",
                    filter="source == 'examples'",
                    output_fields=["id", "title", "url"],
                    limit=100,
                )
                for r in result:
                    resources.append(
                        Resource(
                            uri=r.get("url", "")
                            or f"milvus://{r.get('id', '')}",
                            title=r.get("title", "")
                            or r.get("id", "Unnamed"),
                            description="Stored Milvus document"
                        )
                    )
            else:
                # LangChain Milvus API
                docs: Iterable[Any] = self.milvus_client.similarity_search()
                # todo
        except Exception as e:
            # todo

    def _is_milvus_lite(self) -> bool:
        return DEFAULT_MILVUS_URI.endswith(".db") or (
            not DEFAULT_MILVUS_URI.startswith(("http://", "https://")) and "://" not in DEFAULT_MILVUS_URI
        )

if __name__ == '__main__':
    embed = QwenEmbedding()
    vector = embed.encode_documents(
        [
            "你好"
        ]
    )
    print(vector[0].shape[0])
    milvus = MilvusRetriever()







