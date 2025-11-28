import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
from contextlib import asynccontextmanager
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", "/home/ai/work/models/qwen_embedding_4b")
# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

tokenizer = None
model = None


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model

    try:
        logger.info("正在加载 Qwen3-Embedding 模型...")

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_DIR, padding_side='left', trust_remote_code=True)
        model = AutoModel.from_pretrained(pretrained_model_name_or_path=MODEL_DIR, trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=torch.float16).cuda()
        # model = AutoModel.from_pretrained(pretrained_model_name_or_path=MODEL_DIR, trust_remote_code=True)

        model.to(DEVICE)
        model.eval()

        logger.info(f"模型加载完成！设备: {DEVICE}")

    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise e

    yield

    logger.info("正在清理资源...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("服务关闭完成")


app = FastAPI(
    title="Qwen3 Embedding API",
    description="基于 Qwen3-Embedding的文本嵌入服务",
    version="1.0.0",
    lifespan=lifespan
)


class EmbeddingRequest(BaseModel):
    texts: List[str]
    task_description: Optional[str] = None
    query: Optional[str] = None
    normalize: bool = True
    max_length: int = 8192


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    shape: List[int]


@torch.inference_mode()
def _get_embeddings_batch(texts: List[str], max_length: int = 8192) -> Tensor:
    batch_dict = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch_dict = {k: v.to(model.DEVICE) for k, v in batch_dict.items()}

    outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    return embeddings


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
@torch.inference_mode()
async def get_embeddings(request: EmbeddingRequest):
    if not tokenizer or not model:
        raise HTTPException(status_code=503, detail="模型未加载完成，请稍后重试")

    if not request.texts:
        raise HTTPException(status_code=400, detail="文本列表不能为空")

    try:
        if request.task_description and request.query:
            processed_texts = [get_detailed_instruct(request.task_description, text) for text in request.texts]
        else:
            processed_texts = request.texts

        embeddings = _get_embeddings_batch(processed_texts, request.max_length)

        if request.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        embeddings_list = embeddings.cpu().float().tolist()

        logger.info(f"成功处理 {len(request.texts)} 个文本的嵌入计算")

        return EmbeddingResponse(
            embeddings=embeddings_list,
            shape=[len(embeddings_list), embeddings.shape[1]]
        )

    except Exception as e:
        logger.error(f"嵌入计算失败: {e}")
        raise HTTPException(status_code=500, detail=f"嵌入计算失败: {str(e)}")



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)