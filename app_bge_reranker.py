from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import torch
from typing import List, Any, Optional
from pydantic import BaseModel


MODEL_DIR = os.environ.get("MODEL_DIR", "/home/ai/work/models/bge_reranker_large")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class SentenceInput(BaseModel):
    query: str
    documents: List[str]
    top_n: Optional[int] = None  # default: all
    return_embeddings: bool = False


class ScoresOutput(BaseModel):
    document: str
    relevance_score: float
    index: int
    embedding: Any | None = None

class Output(BaseModel):
    results: List[ScoresOutput]


tokenizer = None
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model
    print("正在加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model.to(DEVICE).eval()
    print("模型加载完成")
    yield
    if model:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("模型已清理")

app = FastAPI(lifespan=lifespan)


@torch.inference_mode()
def score_texts(query: str, documents: List[str], top_n: Optional[int] = None, return_embeddings: bool = False):
    # 处理top_n参数
    if top_n is None:
        top_n = len(documents)
    else:
        top_n = min(top_n, len(documents))

    texts = [[query, doc] for doc in documents]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()

    scores_list = scores.cpu().numpy().tolist()

    results = []
    for i, (doc, score) in enumerate(zip(documents, scores_list)):
        results.append({
            "document": doc,
            "relevance_score": score,
            "index": i,
            "embedding": None
        })

    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    top_results = results[:top_n]

    return top_results


@app.post("/v1/reranker", response_model=Output)
async def calculate_scores(req: SentenceInput):
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="模型未加载完成")

    scs = score_texts(**req.model_dump())
    return Output(results=scs)


# async def main():
#     config = uvicorn.Config(app, host="0.0.0.0", port=5005, log_level="info")
#     server = uvicorn.Server(config)
#     await server.serve()

if __name__ == "__main__":
    import asyncio
    # asyncio.run(main())
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5004)














