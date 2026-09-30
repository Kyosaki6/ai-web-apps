"""Cấu hình tập trung. Mọi giá trị đều ghi đè được bằng biến môi trường."""
import os
from pathlib import Path

# Đảm bảo PyTorch / Tokenizers không xung đột luồng gây EXC_BAD_ACCESS (SIGSEGV) trên macOS ARM
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
if hasattr(torch, "set_num_threads"):
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

ROOT = Path(os.environ.get("APP_ROOT", Path(__file__).resolve().parent))
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Mô hình (đổi tên model = đổi biến môi trường, không sửa code)
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", str(ART_DIR / "detector" / "yolo11n.pt"))
CLIP_MODEL = os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
LLM_MODEL = os.environ.get(
    "LLM_MODEL",
    "Qwen/Qwen2.5-1.5B-Instruct" if DEVICE == "cuda" else "Qwen/Qwen2.5-0.5B-Instruct",
)

# Cấu hình OpenAI-compatible endpoint tùy chọn cho LLM
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "").strip() or None
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip() or "dummy"
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "").strip() or LLM_MODEL

# Bật/tắt từng mô hình để tiết kiệm bộ nhớ, ví dụ ENABLED_MODELS="classifier,detector"
ENABLED_MODELS = {
    m.strip() for m in os.environ.get("ENABLED_MODELS", "classifier,detector,retrieval,llm").split(",") if m.strip()
}

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "8"))
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:5173,http://localhost:8501,https://kyosaki6.github.io",
    ).split(",")
    if origin.strip()
]
PORT = int(os.environ.get("PORT", "8000"))


def resolve_path(path: str) -> Path:
    """Dữ liệu lưu đường dẫn tương đối so với ROOT để mang sang máy khác (Docker, HF Spaces)."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
