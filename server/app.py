from __future__ import annotations

import os
import math
import re
import hmac
import uuid
from functools import lru_cache
from typing import Literal, Optional, Tuple, List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------- Optional deps (light at import) ----------------------------
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None  # type: ignore

# IMPORTANT: defer heavy libs until first use to reduce boot-time memory
AutoTokenizer = None          # will be imported lazily in get_sentiment_runtime()
SentenceTransformer = None    # will be imported lazily in get_embedder()

# ---------------------------- FastAPI ----------------------------
app = FastAPI(title="GhostTab AI API", version=os.getenv("API_VERSION", "1.2.1"))

# ---------------------------- CORS -------------------------------
allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS", "*")
origin_list = [o.strip() for o in allowed_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if origin_list == ["*"] else origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------- Simple API key ------------------------
API_TOKEN = (os.getenv("API_TOKEN") or "").strip()
API_AUTH_DEBUG = os.getenv("API_AUTH_DEBUG", "0") == "1"

def _normalize_uuid_like(s: str) -> str:
    s = (s or "").strip()
    try:
        return str(uuid.UUID(s))
    except Exception:
        return s

async def verify_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="x-api-key"),
    authorization: Optional[str] = Header(default=None),
) -> None:
    if not API_TOKEN:
        return  # auth disabled
    candidate = x_api_key
    if not candidate and authorization:
        auth = authorization.strip()
        if auth.lower().startswith("bearer "):
            candidate = auth.split(" ", 1)[1]
    cand_norm = _normalize_uuid_like(candidate or "")
    exp_norm = _normalize_uuid_like(API_TOKEN)

    if API_AUTH_DEBUG:
        print(f"[auth] received len={len(cand_norm)} {cand_norm[:4]}...{cand_norm[-4:] if cand_norm else ''}")
        print(f"[auth] expected len={len(exp_norm)} {exp_norm[:4]}...{exp_norm[-4:] if exp_norm else ''}")

    if not (exp_norm and hmac.compare_digest(cand_norm, exp_norm)):
        raise HTTPException(status_code=401, detail="Invalid API key")

# --------------------------- Schemas -----------------------------
class InText(BaseModel):
    text: str = Field(min_length=1)

class InRewrite(BaseModel):
    text: str = Field(min_length=1)
    tone: Optional[str] = Field(None, description="Optional tone, e.g., 'friendly', 'assertive'")

class InTranslate(BaseModel):
    text: str = Field(min_length=1)
    to: str = Field(min_length=2, description="Target language code: 'es', 'fr', 'ko', etc.")

class InAskPage(BaseModel):
    text: str
    question: str
    top_k: int | None = 5
    chunk_size: int | None = 1000
    overlap: int | None = 150

class SummaryOut(BaseModel):
    summary: str

class RewriteOut(BaseModel):
    rewrite: str

class TranslateOut(BaseModel):
    translated: str

class SentimentPayload(BaseModel):
    sentiment: Literal["positive", "negative", "neutral", "(stub)"]
    confidence: float = 0.0
    note: Optional[str] = None

class AnalyzeOut(BaseModel):
    sentiment: SentimentPayload
    summary: str

class AskPageOut(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

# --------------------------- LLM Providers (Azure + OpenAI) ------------------------
@lru_cache(maxsize=1)
def get_llm_clients() -> Dict[str, Tuple[object | None, bool]]:
    """
    Returns {"azure": (client, configured), "openai": (client, configured)}.
    """
    try:
        from openai import AzureOpenAI, OpenAI  # type: ignore
    except Exception:
        return {"azure": (None, False), "openai": (None, False)}

    # Azure OpenAI
    azure_key = os.getenv("AZURE_OPENAI_KEY")
    azure_endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-07-18")
    azure_ok = bool(azure_key and azure_endpoint)
    azure_client = AzureOpenAI(api_key=azure_key, azure_endpoint=azure_endpoint, api_version=azure_api_version) if azure_ok else None

    # OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_ok = bool(openai_key)
    openai_client = (None if not openai_ok else __import__("openai").OpenAI(api_key=openai_key))  # late import pattern

    return {"azure": (azure_client, azure_ok), "openai": (openai_client, openai_ok)}

def get_model_name(provider: Optional[str]) -> Optional[str]:
    """
    For 'azure', return the deployment name (AZURE_OPENAI_DEPLOYMENT).
    For 'openai', return the model name (OPENAI_MODEL).
    """
    if provider == "azure":
        return os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    if provider == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-5-mini")
    return None

def pick_provider(x_llm: Optional[str] = Header(default=None, alias="x-llm")) -> Optional[str]:
    """
    Header-based provider selector. Returns 'azure', 'openai', or None for auto.
    """
    v = (x_llm or "").strip().lower()
    if v in {"azure", "openai"}:
        return v
    return None

def _chat_complete(prompt: str, provider: Optional[str] = None) -> str:
    """
    provider: 'azure' | 'openai' | None (auto: prefer azure if configured, else openai)
    """
    clients = get_llm_clients()
    if provider not in {"azure", "openai"}:
        provider = "azure" if clients["azure"][1] else ("openai" if clients["openai"][1] else None)
    if provider is None:
        return "(stub) No LLM credentials configured."

    client, ok = clients[provider]
    if not ok or client is None:
        return "(stub) No LLM credentials configured."

    model = get_model_name(provider)
    if not model:
        raise HTTPException(status_code=500, detail=f"Model/deployment not configured for {provider}")

    try:
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if os.getenv("ALLOW_TEMPERATURE", "false").lower() == "true":
            kwargs["temperature"] = 0.2
        # both AzureOpenAI and OpenAI follow the same .chat.completions.create API
        resp = client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or ""
        return content.strip()
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------- ONNX Sentiment -----------------------
BASE_DIR = os.path.dirname(__file__)
SENTIMENT_DIR = os.path.join(BASE_DIR, "models", "sentiment")
SENTIMENT_ONNX = os.path.join(SENTIMENT_DIR, "model.onnx")

@lru_cache(maxsize=1)
def get_sentiment_runtime():
    """
    Initializes and caches ONNX runtime & tokenizer lazily.
    Returns (available, session, tokenizer, note)
    """
    if ort is None:
        return False, None, None, "onnxruntime not installed"
    if not os.path.exists(SENTIMENT_ONNX):
        return False, None, None, "ONNX model missing. Run scripts/export_sentiment.py"

    # Lazy import transformers' AutoTokenizer
    global AutoTokenizer
    if AutoTokenizer is None:
        try:
            from transformers import AutoTokenizer as _AT  # type: ignore
            AutoTokenizer = _AT
        except Exception:
            return False, None, None, "transformers not installed"

    try:
        session = ort.InferenceSession(SENTIMENT_ONNX, providers=["CPUExecutionProvider"])
        tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_DIR)
        return True, session, tokenizer, None
    except Exception as e:  # pragma: no cover
        return False, None, None, str(e)

# --------------------------- RAG Utilities -----------------------
def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> List[Dict[str, Any]]:
    text = re.sub(r"\s+\n", "\n", text).strip()
    chunks: List[Dict[str, Any]] = []
    i = 0
    idx = 0
    L = len(text)
    while i < L:
        j = min(i + size, L)
        chunk = text[i:j]
        chunks.append({"idx": idx, "start": i, "end": j, "text": chunk})
        idx += 1
        if j == L:
            break
        i = max(0, j - overlap)
    return chunks

@lru_cache(maxsize=1)
def get_embedder() -> Optional["SentenceTransformer"]:
    """Load a small, CPU-friendly embedding model (cached, lazy)."""
    global SentenceTransformer
    if SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer as _ST  # type: ignore
            SentenceTransformer = _ST
        except Exception:
            return None
    model_name = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    try:
        return SentenceTransformer(model_name, device="cpu")
    except Exception:
        return None

def embed_texts(texts: List[str]) -> np.ndarray:
    if np is None:
        raise HTTPException(status_code=500, detail="NumPy not available for embeddings")
    embedder = get_embedder()
    if embedder is None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(len(texts), 384)).astype(np.float32)
    else:
        X = np.asarray(embedder.encode(texts, batch_size=32, show_progress_bar=False), dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return X / norms

def top_k_by_cosine(query_vec: np.ndarray, doc_mat: np.ndarray, k: int = 5) -> List[int]:
    sims = doc_mat @ query_vec.reshape(-1, 1)
    order = np.argsort(-sims.squeeze())
    return order[:k].tolist()

def make_cited_prompt(question: str, chunk_texts: List[str]) -> str:
    numbered = "\n\n".join([f"[{i+1}] {t}" for i, t in enumerate(chunk_texts)])
    return (
        "You are a concise assistant. Answer ONLY using the provided context. "
        "If the answer is not present, say you don't know.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{numbered}\n\n"
        "Answer with citations like [1], [2] when relevant."
    )

# ---------------------------- Routes -----------------------------
# ----- NEW PUBLIC /ping (no auth) -----
@app.get("/ping")
def ping():
    """Public, fast health probe for the extension."""
    clients = get_llm_clients()
    llm_mode = "azure" if clients["azure"][1] else ("openai" if clients["openai"][1] else None)
    return {
        "ok": True,
        "service": "ghosttab-api",
        "version": app.version,
        "llm_mode": llm_mode,
    }

@app.get("/")
def root():
    available, _, _, note = get_sentiment_runtime()
    # keep legacy llm_mode for compatibility (first configured provider)
    clients = get_llm_clients()
    llm_mode = "azure" if clients["azure"][1] else ("openai" if clients["openai"][1] else None)
    return {
        "ok": True,
        "service": "ghosttab-api",
        "version": app.version,
        "sentiment_model": bool(available),
        "sentiment_note": note,
        "llm_mode": llm_mode,
    }

@app.get("/health")
def health():
    available, _, _, _ = get_sentiment_runtime()
    clients = get_llm_clients()
    llm_any = bool(clients["azure"][1] or clients["openai"][1])
    return {
        "status": "healthy",
        "sentiment_available": bool(available),
        "llm_configured": llm_any,
        "providers": {
            "azure": clients["azure"][1],
            "openai": clients["openai"][1],
        },
    }

# ----- Debug echo -----
@app.get("/_debug/echo")
def echo(x_api_key: str | None = Header(default=None, alias="x-api-key"),
         authorization: str | None = Header(default=None)):
    def shape(v: str | None):
        v = (v or "").strip()
        return {
            "present": bool(v),
            "len": len(v),
            "head": v[:4],
            "tail": v[-4:] if v else "",
        }
    return {
        "x-api-key": shape(x_api_key),
        "authorization": shape(authorization),
        "api_token_on_server": {"present": bool(API_TOKEN), "len": len(API_TOKEN)},
        "version": app.version,
    }

# ------------------------- Helpers -------------------------------
def _sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except TypeError:
        return 1 / (1 + np.exp(-x))  # type: ignore

# ---------------------------- LLM Ops ----------------------------
@app.post("/summarize", response_model=SummaryOut, dependencies=[Depends(verify_api_key)])
async def summarize(inp: InText, provider: Optional[str] = Depends(pick_provider)):
    text = inp.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    prompt = f"Summarize this in 5 clear, concise bullet points:\n\n{text[:8000]}"
    summary = _chat_complete(prompt, provider=provider)
    return SummaryOut(summary=summary)

@app.post("/rewrite", response_model=RewriteOut, dependencies=[Depends(verify_api_key)])
async def rewrite(inp: InRewrite, provider: Optional[str] = Depends(pick_provider)):
    text = inp.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    tone = (inp.tone or "").strip()
    style = f" in a {tone} tone" if tone else ""
    prompt = f"Rewrite the following text to be concise and professional{style}:\n\n{text[:4000]}"
    rewrite_txt = _chat_complete(prompt, provider=provider)
    return RewriteOut(rewrite=rewrite_txt)

@app.post("/translate", response_model=TranslateOut, dependencies=[Depends(verify_api_key)])
async def translate(inp: InTranslate, provider: Optional[str] = Depends(pick_provider)):
    text = inp.text.strip()
    lang = inp.to.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    if not lang:
        raise HTTPException(status_code=400, detail="No target language provided")
    prompt = f"Translate this into {lang}:\n\n{text[:4000]}"
    translated = _chat_complete(prompt, provider=provider)
    return TranslateOut(translated=translated)

# ------------------------- Sentiment Ops -------------------------
@app.post("/sentiment", response_model=SentimentPayload, dependencies=[Depends(verify_api_key)])
async def sentiment(inp: InText):
    text = inp.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    available, session, tokenizer, note = get_sentiment_runtime()
    if not available or session is None or tokenizer is None:
        return SentimentPayload(sentiment="(stub)", confidence=0.0, note=note)

    toks = tokenizer([text], return_tensors="np", padding=True, truncation=True, max_length=256)
    ort_inputs = {"input_ids": toks["input_ids"], "attention_mask": toks["attention_mask"]}
    logits = session.run(None, ort_inputs)[0]

    if np is None:  # pragma: no cover
        raw0, raw1 = float(logits[0][0]), float(logits[0][1])
        ex0, ex1 = math.exp(raw0), math.exp(raw1)
        p0, p1 = ex0 / (ex0 + ex1), ex1 / (ex0 + ex1)
        probs = [p0, p1]
    else:
        sig = 1 / (1 + np.exp(-logits))
        row = sig[0].astype(float)
        s = float(row.sum()) or 1.0
        probs = (row / s).tolist()

    pred_idx = 1 if probs[1] >= probs[0] else 0
    label = "positive" if pred_idx == 1 else "negative"
    confidence = float(probs[pred_idx])

    if confidence < float(os.getenv("SENTIMENT_NEUTRAL_THRESH", 0.60)):
        label = "neutral"

    return SentimentPayload(sentiment=label, confidence=round(confidence, 4))

# -------- NEW: /analyze (runs ONNX + GPT summary in one call) ----
@app.post("/analyze", response_model=AnalyzeOut, dependencies=[Depends(verify_api_key)])
async def analyze(inp: InText, provider: Optional[str] = Depends(pick_provider)):
    s = await sentiment(inp)
    try:
        text = inp.text.strip()
        prompt = f"Summarize this in 5 clear, concise bullet points:\n\n{text[:8000]}"
        summary = _chat_complete(prompt, provider=provider)
    except HTTPException:
        summary = "(stub) No LLM credentials configured."
    return AnalyzeOut(sentiment=s, summary=summary)

# -------- NEW: /ask_page (RAG pipeline) -------------------------
@app.post("/ask_page", response_model=AskPageOut, dependencies=[Depends(verify_api_key)])
async def ask_page(inp: InAskPage, provider: Optional[str] = Depends(pick_provider)):
    page_text = (inp.text or "").strip()
    q = (inp.question or "").strip()
    if not page_text:
        raise HTTPException(status_code=400, detail="No page text provided")
    if not q:
        raise HTTPException(status_code=400, detail="No question provided")

    chunks = chunk_text(page_text, size=inp.chunk_size or 1000, overlap=inp.overlap or 150)
    if not chunks:
        return AskPageOut(answer="I don't know.", sources=[])

    chunk_vecs = embed_texts([c["text"] for c in chunks])
    query_vec = embed_texts([q])[0]
    k = max(1, min(inp.top_k or 5, len(chunks)))
    idxs = top_k_by_cosine(query_vec, chunk_vecs, k=k)
    top_chunks = [chunks[i] for i in idxs]

    prompt = make_cited_prompt(q, [c["text"] for c in top_chunks])

    try:
        answer = _chat_complete(prompt, provider=provider)
    except HTTPException:
        answer = "(stub) No LLM credentials configured."

    sources = [
        {"rank": r + 1, "chunk_idx": c["idx"], "start": c["start"], "end": c["end"], "preview": c["text"][:220]}
        for r, c in enumerate(top_chunks)
    ]
    return AskPageOut(answer=answer, sources=sources)

# ----------------------- Error Handlers --------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

# ----------------------- Startup Log (safe) ----------------------
@app.on_event("startup")
async def _startup_log():
    print(f"[boot] API_TOKEN present={bool(API_TOKEN)} len={len(API_TOKEN)} version={app.version}")
