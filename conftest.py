"""Hạ tầng test: chạy `pytest -q` từ repo root là đủ.

- Đảm bảo repo root nằm trên sys.path để `from api import main` luôn đúng.
- Nếu máy chưa cài torch (model thật được mock nên không cần), dựng stub
  tối thiểu cho `config.py`. Khi đã cài torch thật, stub không dùng tới.
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch  # noqa: F401
except ImportError:
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = torch
