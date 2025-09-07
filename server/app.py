from __future__ import annotations

import os
import math
from functools import lru_cache
from typing import Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Optional deps — we handle absence gracefully
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None  # type: ignore

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    AutoTokenizer = None  # type: ignore

# ---------------------------- FastAPI ----------------------------
app = FastAPI(title="GhostTab AI API", version=os.getenv("API_VERSION", "1.1.0"))

# ---------------------------- CORS -------------------------------
# Prefer explicit origins in prod via CORS_ALLOWED_ORIGINS env
# e.g. "chrome-extension://abc123,https://ghosttab.app"
allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS", "*")
origin_list = [o.strip() for o in allowed_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if origin_list == ["*"] else origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------- Schemas -----------------------------
class InText(BaseModel):
    text: str = Field(min_length=1)


class InRewrite(BaseModel):
    text: str = Field(min_length=1)
    tone: Optional[str] = Field(None, description="Optional tone, e.g., 'friendly', 'assertive'")


class InTranslate(BaseModel):
    text: str = Field(min_length=1)
    to: str = Field(min_length=2, description="Target language code: 'es', 'fr', 'ko', etc.")


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


# --------------------------- OpenAI Client ------------------------
# We support either Azure OpenAI or OpenAI — auto-detected from env
# Return (client, mode) where mode in {"azure", "openai"}
@lru_cache(maxsize=1)
def get_llm_client() -> Tuple[object | None, Optional[str]]:
    try:
        from openai import AzureOpenAI, OpenAI  # type: ignore
    except Exception:
        return None, None

    # Prefer Azure OpenAI if endpoint + key exist
    azure_key = os.getenv("AZURE_OPENAI_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-07-18")
    if azure_key and azure_endpoint:
        client = AzureOpenAI(
            api_key=azure_key,
            azure_endpoint=azure_endpoint.rstrip("/"),
            api_version=azure_api_version,
        )
        return client, "azure"

    # Fallback: Regular OpenAI (optional)
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        client = OpenAI(api_key=openai_key)
        return client, "openai"

    # No credentials → stub mode
    return None, None


def get_model_name(mode: Optional[str]) -> Optional[str]:
    """
    For Azure, return the *deployment name* (not base model).
    For OpenAI, return the model name.
    Defaults are sensible given the user's pricing/model access.
    """
    if mode == "azure":
        return os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    if mode == "openai":
        # Prefer a fast & affordable default; override with OPENAI_MODEL
        return os.getenv("OPENAI_MODEL", "gpt-5-mini")
    return None


# --------------------------- ONNX Sentiment -----------------------
BASE_DIR = os.path.dirname(__file__)
SENTIMENT_DIR = os.path.join(BASE_DIR, "models", "sentiment")
SENTIMENT_ONNX = os.path.join(SENTIMENT_DIR, "model.onnx")

@lru_cache(maxsize=1)
def get_sentiment_runtime():
    """Initializes and caches ONNX runtime & tokenizer.

    Returns (available, session, tokenizer, note)
    """
    if ort is None or AutoTokenizer is None:
        return False, None, None, "onnxruntime/transformers not installed"

    if not os.path.exists(SENTIMENT_ONNX):
        return False, None, None, "ONNX model missing. Run scripts/export_sentiment.py"

    try:
        session = ort.InferenceSession(
            SENTIMENT_ONNX,
            providers=["CPUExecutionProvider"],
        )
        tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_DIR)
        return True, session, tokenizer, None
    except Exception as e:  # pragma: no cover
        return False, None, None, str(e)


# ---------------------------- Routes -----------------------------
@app.get("/")
def root():
    available, _, _, note = get_sentiment_runtime()
    return {
        "ok": True,
        "service": "ghosttab-api",
        "version": app.version,
        "sentiment_model": bool(available),
        "sentiment_note": note,
        "llm_mode": get_llm_client()[1],
    }


@app.get("/health")
def health():
    available, _, _, _ = get_sentiment_runtime()
    client, mode = get_llm_client()
    return {
        "status": "healthy",
        "sentiment_available": bool(available),
        "llm_configured": bool(client and get_model_name(mode)),
    }


# ------------------------- Helpers -------------------------------

def _chat_complete(prompt: str) -> str:
    client, mode = get_llm_client()

    if client is None:
        # Stub mode for local/dev without keys
        return "(stub) No LLM credentials configured."

    model = get_model_name(mode)
    if not model:
        raise HTTPException(status_code=500, detail="Model/deployment not configured")

    try:
        resp = client.chat.completions.create(  # type: ignore[attr-defined]
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = resp.choices[0].message.content or ""
        return content.strip()
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(e))


def _sigmoid(x):
    # Robust sigmoid for both Python float and numpy arrays
    try:
        return 1 / (1 + math.exp(-x))
    except TypeError:
        # Fallback for numpy arrays
        return 1 / (1 + np.exp(-x))  # type: ignore


# ---------------------------- LLM Ops ----------------------------
@app.post("/summarize", response_model=SummaryOut)
async def summarize(inp: InText):
    text = inp.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    prompt = f"Summarize this in 5 clear, concise bullet points:\n\n{text[:8000]}"
    summary = _chat_complete(prompt)
    return SummaryOut(summary=summary)


@app.post("/rewrite", response_model=RewriteOut)
async def rewrite(inp: InRewrite):
    text = inp.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    tone = (inp.tone or "").strip()
    style = f" in a {tone} tone" if tone else ""
    prompt = f"Rewrite the following text to be concise and professional{style}:\n\n{text[:4000]}"
    rewrite_txt = _chat_complete(prompt)
    return RewriteOut(rewrite=rewrite_txt)


@app.post("/translate", response_model=TranslateOut)
async def translate(inp: InTranslate):
    text = inp.text.strip()
    lang = inp.to.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    if not lang:
        raise HTTPException(status_code=400, detail="No target language provided")

    prompt = f"Translate this into {lang}:\n\n{text[:4000]}"
    translated = _chat_complete(prompt)
    return TranslateOut(translated=translated)


# ------------------------- Sentiment Ops -------------------------
@app.post("/sentiment", response_model=SentimentPayload)
async def sentiment(inp: InText):
    text = inp.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    available, session, tokenizer, note = get_sentiment_runtime()
    if not available or session is None or tokenizer is None:
        return SentimentPayload(sentiment="(stub)", confidence=0.0, note=note)

    # Tokenize
    toks = tokenizer(
        [text],
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=256,
    )

    # Prepare ONNX inputs
    ort_inputs = {
        "input_ids": toks["input_ids"],
        "attention_mask": toks["attention_mask"],
    }

    # Run inference
    logits = session.run(None, ort_inputs)[0]

    # Convert to probabilities (binary case assumed)
    if np is None:  # pragma: no cover
        # Minimal fallback if numpy is missing
        raw0, raw1 = float(logits[0][0]), float(logits[0][1])
        # Normalize with softmax-like approach
        ex0, ex1 = math.exp(raw0), math.exp(raw1)
        p0, p1 = ex0 / (ex0 + ex1), ex1 / (ex0 + ex1)
        probs = [p0, p1]
    else:
        # Sigmoid per-logit then normalize (robust for binary outputs)
        sig = 1 / (1 + np.exp(-logits))
        row = sig[0].astype(float)
        s = float(row.sum()) or 1.0
        probs = (row / s).tolist()

    pred_idx = 1 if probs[1] >= probs[0] else 0
    label = "positive" if pred_idx == 1 else "negative"
    confidence = float(probs[pred_idx])

    # Add "neutral" if confidence is too low
    if confidence < float(os.getenv("SENTIMENT_NEUTRAL_THRESH", 0.60)):
        label = "neutral"

    return SentimentPayload(sentiment=label, confidence=round(confidence, 4))


# -------- NEW: /analyze (runs ONNX + GPT summary in one call) ----
@app.post("/analyze", response_model=AnalyzeOut)
async def analyze(inp: InText):
    # 1) ONNX sentiment (never throws; may return stub)
    s = await sentiment(inp)

    # 2) GPT summary (may be stub if no keys)
    try:
        summ = await summarize(inp)
        summary = summ.summary
    except HTTPException:
        summary = "(stub) No LLM credentials configured."

    return AnalyzeOut(sentiment=s, summary=summary)


# ----------------------- Error Handlers --------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    return fastapi.responses.JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )
