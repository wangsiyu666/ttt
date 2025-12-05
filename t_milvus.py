from __future__ import annotations
import shutil
import tempfile
from types import SimpleNamespace
from uuid import uuid4
from retriever import Resource, Document
from milvus import MilvusRetriever
from retriever import Retriever
from retriever_tools import get_retriever_tool

# tool = get_retriever_tool(
#     [{
#         "uri": "milvus://test/deer-flow.md",
#         "title": "deer-flow",
#         "description": "deer-flow",
#     }]
# )
# r = tool._run("deer-flow讲的什么？")
# print(r)
milvus = MilvusRetriever()
# milvus.load_examples()
r = milvus.query_relevant_documents(
    "vanna是什么？",
    [
        Resource(**{
            "uri": "example_deer-flow_e81fab1d",
            "title": "SQLAgent",
            "description": "SQLAgent"
        })
    ]
)
print(r)











