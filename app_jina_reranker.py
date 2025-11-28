from transformers import AutoModel
import torch
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from typing import List, Optional, Any
from pydantic import BaseModel
import numpy as np

MODEL_DIR = os.environ.get("MODEL_DIR", "/home/ai/work/models/jina-reranker-v3")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class SentenceInput(BaseModel):
    query: str
    documents: List[str]
    top_n: Optional[int] = 3  # default: all
    return_embeddings: bool = False

class ScoresOutput(BaseModel):
    document: str
    relevance_score: float
    index: int
    embedding: Any | None = None

class Output(BaseModel):
    results: List[ScoresOutput]

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("正在加载模型")
    model = AutoModel.from_pretrained(
        MODEL_DIR,
        dtype="auto",
        trust_remote_code=True,
    )
    model.to(DEVICE).eval()
    print(f"模型加载完成: {DEVICE}")
    yield
    if model:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("模型已清理")

app = FastAPI(lifespan=lifespan)

@torch.inference_mode()
def score_texts(query: str, documents: List[str], top_n=3, return_embeddings=False):
    top_n = min(top_n, len(documents))
    top_results = model.rerank(query, documents, top_n=top_n, return_embeddings=return_embeddings)
    processed_results = []
    for result in top_results:
        processed_result = {
            "document": result["document"],
            "relevance_score": float(result["relevance_score"]),  # 确保是 Python float
            "index": result["index"],
            "embedding": None
        }

        if return_embeddings and "embedding" in result and result["embedding"] is not None:
            if isinstance(result["embedding"], np.ndarray):
                processed_result["embedding"] = result["embedding"].tolist()
            else:
                processed_result["embedding"] = result["embedding"]

        processed_results.append(processed_result)

    return processed_results

@app.post("/v1/reranker", response_model=Output)
async def calculate_scores(req: SentenceInput):
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载完成")

    scs = score_texts(**req.model_dump())
    return Output(results=scs)

# async def main():
#     config = uvicorn.Config(app, host="0.0.0.0", port=5005, log_level="info")
#     server = uvicorn.Server(config)
#     await server.serve()

if __name__ == '__main__':
    import asyncio
    # asyncio.run(main())

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5005)

















