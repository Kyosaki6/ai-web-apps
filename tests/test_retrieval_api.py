"""QA-01: test 3 endpoint retrieval (search/text, search/image, gallery) bằng FakeSearch.

Không nạp CLIP, không cần GPU. TestClient KHÔNG dùng `with`, nên lifespan
(_load_models) không chạy. Chạy từ repo root: pip install pytest httpx && pytest -q
"""
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import main


class FakeSearch:
    """Thay ImageSearch thật: cùng giao diện (meta, search_text, search_image)."""

    def __init__(self, n=3):
        self.meta = [{"path": f"data/gallery/{i}.png", "label": f"flower{i}"} for i in range(n)]

    def _hits(self, k):
        return [
            {"id": i, "score": round(1.0 - i * 0.1, 2), "label": m["label"], "path": m["path"]}
            for i, m in enumerate(self.meta[:k])
        ]

    def search_text(self, query, k=8):
        return self._hits(k)

    def search_image(self, image, k=8):
        assert isinstance(image, Image.Image)
        return self._hits(k)


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(buf, format="PNG")
    return buf.getvalue()


IMG = {"file": ("q.png", png_bytes(), "image/png")}


@pytest.fixture(scope="session")
def gallery_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("gallery")
    for i in range(3):
        (d / f"{i}.png").write_bytes(png_bytes())
    return d


@pytest.fixture()
def client(gallery_dir, monkeypatch):
    monkeypatch.setattr(main, "MODELS", {"retrieval": FakeSearch()})
    # meta lưu path tương đối; đổi sang thư mục tạm khi test
    monkeypatch.setattr(main, "resolve_path", lambda p: gallery_dir / Path(p).name)
    return TestClient(main.app)


# ---------- POST /api/search/text ----------
def test_text_ok(client):
    r = client.post("/api/search/text", json={"query": "red rose", "k": 2})
    assert r.status_code == 200
    res = r.json()["results"]
    assert len(res) == 2
    assert res[0]["url"] == "/api/gallery/0"
    assert res[0]["label"] == "flower0"
    assert "path" not in res[0]  # không lộ đường dẫn nội bộ


def test_text_default_k(client):
    r = client.post("/api/search/text", json={"query": "daisy"})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 3  # gallery giả chỉ có 3 ảnh


@pytest.mark.parametrize(
    "body",
    [
        {"k": 3},                        # thiếu query
        {"query": "x", "k": "abc"},      # sai kiểu k
        {"query": "x", "k": 0},          # k < 1
        {"query": "x", "k": 25},         # k > 24
        {"query": ""},                   # query rỗng
        {"query": "x" * 201},            # query quá dài
    ],
)
def test_text_invalid_422(client, body):
    assert client.post("/api/search/text", json=body).status_code == 422


# ---------- POST /api/search/image ----------
def test_image_ok(client):
    r = client.post("/api/search/image", files=IMG, data={"k": "2"})
    assert r.status_code == 200
    res = r.json()["results"]
    assert len(res) == 2 and res[0]["url"] == "/api/gallery/0"


def test_image_k_is_clamped(client):
    r = client.post("/api/search/image", files=IMG, data={"k": "999"})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 3


def test_image_missing_file_422(client):
    # File(...) bắt buộc: FastAPI tự trả 422 khi thiếu trường file
    assert client.post("/api/search/image").status_code == 422


def test_image_not_an_image_400(client):
    r = client.post("/api/search/image", files={"file": ("q.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_image_too_large_413(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_MB", 0)
    assert client.post("/api/search/image", files=IMG).status_code == 413


def test_image_bad_k_type_422(client):
    assert client.post("/api/search/image", files=IMG, data={"k": "abc"}).status_code == 422


# ---------- GET /api/gallery/{id} ----------
def test_gallery_ok(client):
    r = client.get("/api/gallery/1")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/")


@pytest.mark.parametrize("item_id", [3, 999, -1])
def test_gallery_not_found_404(client, item_id):
    assert client.get(f"/api/gallery/{item_id}").status_code == 404


def test_gallery_bad_id_422(client):
    assert client.get("/api/gallery/abc").status_code == 422


# ---------- retrieval không được nạp (tắt trong ENABLED_MODELS hoặc nạp lỗi) -> 503 ----------
@pytest.mark.parametrize(
    "method,url,kwargs",
    [
        ("post", "/api/search/text", {"json": {"query": "x"}}),
        ("post", "/api/search/image", {"files": IMG}),
        ("get", "/api/gallery/0", {}),
    ],
)
def test_model_not_loaded_503(client, monkeypatch, method, url, kwargs):
    monkeypatch.setattr(main, "MODELS", {})
    assert getattr(client, method)(url, **kwargs).status_code == 503
