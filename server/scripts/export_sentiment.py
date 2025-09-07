import os
from pathlib import Path
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------- Config --------------------
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "models" / "sentiment"
ONNX_PATH = OUT_DIR / "model.onnx"

# Safer ONNX export on recent PyTorch
os.environ.setdefault("PYTORCH_ONNX_DISABLE_SDPA", "1")

def export_model(opset: int = 17, max_len: int = 128):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("📥 Downloading model + tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    print(f"💾 Saving tokenizer to: {OUT_DIR}")
    tokenizer.save_pretrained(str(OUT_DIR))

    print(f"🔄 Exporting to ONNX (opset={opset}, max_len={max_len})…")
    # Use padded sample so exporter sees both batch & sequence dims
    sample = tokenizer(
        "Export test",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_len,
    )

    with torch.no_grad():
        torch.onnx.export(
            model,
            (sample["input_ids"], sample["attention_mask"]),
            str(ONNX_PATH),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            # Make BOTH batch (0) and sequence (1) dynamic
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            },
            opset_version=opset,
        )

    print(f"✅ Export complete: {ONNX_PATH}")
    return tokenizer

def try_verify(tokenizer):
    """Optional: verify ONNX with a tiny inference if onnxruntime is available."""
    try:
        import onnxruntime as ort
        import numpy as np
    except Exception as e:
        print(f"ℹ️ Skipping verification (onnxruntime not available): {e}")
        return

    print("🧪 Verifying ONNX with a quick inference…")
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    toks = tokenizer(
        ["I absolutely love GhostTab!"],
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=128,  # can be any <= what your runtime uses
    )
    inputs = {"input_ids": toks["input_ids"], "attention_mask": toks["attention_mask"]}
    logits = sess.run(None, inputs)[0]
    probs = 1 / (1 + np.exp(-logits))
    pred = int(probs.argmax(axis=1)[0])
    label = "positive" if pred == 1 else "negative"
    conf = float(probs[0, pred])
    print(f"✅ Verification OK → {label} (conf={conf:.4f})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (>=14). Default: 17")
    parser.add_argument("--max_len", type=int, default=128, help="Sample max length used during export.")
    args = parser.parse_args()

    tokenizer = export_model(args.opset, args.max_len)
    try_verify(tokenizer)

if __name__ == "__main__":
    main()
