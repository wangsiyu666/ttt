from contextlib import asynccontextmanager
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List

MODEL_DIR = os.environ.get("MODEL_DIR", "/home/ai/work/models/bge")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

class SentenceInput(BaseModel):
    inputs: List[str]

class EncodingOutput(BaseModel):
    embeddings: list[list[float]]

# 全局变量
tokenizer = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model
    print("正在加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(pretrained_model_name_or_path=MODEL_DIR, trust_remote_code=True)
    model.to(DEVICE).eval()
    print("模型加载完成！")
    yield
    if model:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("模型已清理！")

app = FastAPI(lifespan=lifespan)

@torch.inference_mode()
def encode_texts(texts: List[str]):
    encoded_input = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors='pt',
    ).to(DEVICE)

    model_output = model(**encoded_input)
    sentence_embeddings = model_output[0][:, 0]
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.cpu().numpy().tolist()

@app.post("/v1/embeddings", response_model=EncodingOutput)
async def encode_sentences(req: SentenceInput):
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="模型未加载完成")

    embs = encode_texts(req.inputs)
    return EncodingOutput(embeddings=embs)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5001)