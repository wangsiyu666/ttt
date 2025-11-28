"""
    CUDA_VISIBLE_DEVICES=-1 MODEL_DIR=/home/ai/work/models/jina-embeddings-v4-vllm-retrieval /home/ai/anaconda3/envs/jina/bin/python -m uvicorn app_jina_embedding_v4:app --host 0.0.0.0 --port 8898
"""
import os, torch, base64, io
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoProcessor, AutoModel
from contextlib import asynccontextmanager


MODEL_DIR = os.environ.get("MODEL_DIR", "/home/ai/work/models/jina-embeddings-v4-vllm-retrieval")
# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True)
model.to(DEVICE).eval()

processor = None
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, model

    print("正在加载 Jina v4 模型...")
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model.to(DEVICE).eval()
    print("模型加载完成！")
    yield

    print("正在清理资源...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("服务关闭完成")

app = FastAPI(
    title="Jina v4 Multimodal Embeddings (Transformers)",
    lifespan=lifespan
)


class MMInput(BaseModel):
    text: str | None = None
    image_base64: str | None = None

class MMRequest(BaseModel):
    inputs: list[MMInput]
    normalize: bool = True
    pooling: str = "mean"

class MMResponse(BaseModel):
    embeddings: list[list[float]]

def _load_image_from_b64(s: str) -> Image.Image:
    if "," in s and s.strip().startswith("data:"):
        s = s.split(",", 1)[1]
    img_bytes = base64.b64decode(s)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

@torch.inference_mode()
def encode_one(item: MMInput, pooling: str = "mean", normalize: bool = True):
    image = _load_image_from_b64(item.image_base64) if item.image_base64 else None

    if image is not None:
        if not item.text:
            text_input = "<|vision_start|><|image_pad|><|vision_end|>"
        else:
            if "<|image_pad|>" not in item.text and "<|vision_start|>" not in item.text:
                text_input = f"<|vision_start|><|image_pad|><|vision_end|>{item.text}"
            else:
                text_input = item.text
    else:
        text_input = item.text or ""

    processor_kwargs = {
        "text": [text_input],
        "return_tensors": "pt",
        "padding": True,
    }
    
    if image is not None:
        processor_kwargs["images"] = [image]
    
    inputs = processor(**processor_kwargs)

    if "input_ids" in inputs:
        inputs["input_ids"] = inputs["input_ids"].long()
    if "attention_mask" in inputs:
        inputs["attention_mask"] = inputs["attention_mask"].long()

    if image is not None:
        num_image_tokens = (inputs["input_ids"] == 151655).sum().item()
        print(f"Debug: Text='{text_input[:80]}...', Image tokens={num_image_tokens}")
    
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    outputs = model(**inputs)
    last_hidden = outputs.last_hidden_state
    emb = last_hidden.mean(dim=1) if pooling == "mean" else last_hidden[:, 0, :]

    if normalize:
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)

    return emb[0].to("cpu").float().tolist()

@app.post("/v1/embeddings", response_model=MMResponse)
def mm_embeddings(req: MMRequest):
    embs = [encode_one(x, req.pooling, req.normalize) for x in req.inputs]
    return MMResponse(embeddings=embs)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5002)