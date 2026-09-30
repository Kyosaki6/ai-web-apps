# AI Web Apps — Streamlit & React

Bốn ứng dụng AI (phân loại ảnh, phát hiện đối tượng, tìm kiếm ảnh, chatbot RAG) sau một backend FastAPI,
với hai giao diện: Streamlit và React.

## Chạy trên máy (Python 3.11, Node 22)
```bash
pip install -r requirements.txt                       # CPU: cài torch bản CPU trước (xem Dockerfile)
# File nặng (*.pt, *.faiss, data/gallery, data/flowers, data/coco128) không push git:
bash scripts/download_artifacts.sh                    # điền link Drive/HF Release trước khi chạy

uvicorn api.main:app --port 8000 --reload             # backend; mở http://localhost:8000/docs (Swagger)
API_URL=http://localhost:8000 streamlit run streamlit_app.py  # Streamlit (API-only client): http://localhost:8501
cd web && npm install && npm run dev                  # React dev: http://localhost:5173 (proxy /api → :8000)
cd web && npm run build                               # build xong, FastAPI phục vụ luôn React ở http://localhost:8000/
ENABLED_MODELS= pytest -q                             # test mocked, không cần GPU/model thật
```
Máy yếu: `ENABLED_MODELS=classifier,detector uvicorn api.main:app --port 8000` để nạp ít model.

## Biến môi trường (xem `config.py`, mẫu ở `.env.example`)
| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `ENABLED_MODELS` | `classifier,detector,retrieval,llm` | Mô hình được nạp |
| `LLM_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` (GPU) / `-0.5B-` (CPU) | Mô hình sinh |
| `EMBED_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Embedding cho RAG |
| `CLIP_MODEL` | `openai/clip-vit-base-patch32` | Tìm kiếm ảnh |
| `YOLO_WEIGHTS` | `artifacts/detector/yolo11n.pt` | Weights YOLO |
| `OPENAI_API_BASE` / `OPENAI_API_KEY` / `OPENAI_MODEL` | _(trống)_ / `dummy` / = `LLM_MODEL` | Endpoint OpenAI-compatible tùy chọn cho chat |
| `MAX_UPLOAD_MB` | `8` | Giới hạn ảnh upload |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:8501` | Origin được gọi API |
| `PORT` | `8000` | Cổng backend |
| `API_URL` | `http://localhost:8000` | (Streamlit) địa chỉ backend |
| `OPENAI_API_BASE` | _(trống = model local)_ | Base URL OpenAI-compatible, ví dụ `https://api.freetheai.xyz/v1` (có/không `/v1` đều được) |
| `OPENAI_API_KEY` | `dummy` | API key của provider (FreeTheAI: key `fta_...` từ Discord `/signup`, không commit) |
| `OPENAI_MODEL` | `= LLM_MODEL` | Alias model trên provider, ví dụ `glm/glm-5.1` (lấy từ `GET /api/chat/models`, KHÔNG dùng tên HuggingFace) |

## Chatbot RAG qua FreeTheAI
```bash
export OPENAI_API_BASE=https://api.freetheai.xyz/v1
export OPENAI_API_KEY=fta_...            # Discord /signup; mỗi ngày UTC chạy /checkin một lần
export OPENAI_MODEL=glm/glm-5.1          # alias trong https://freetheai.xyz/models
uvicorn api.main:app --port 8000
```
Lỗi thường gặp: `401` = sai key (`/resetkey` để cấp lại), `403 daily_checkin_required` = chưa `/checkin`
hôm nay, `403 model_access_denied` = alias cần role `seems_legit`, `400 unknown aliased model` =
sai alias (kiểm tra `GET /api/chat/models`).

## API (Swagger: `http://localhost:8000/docs`)
| Endpoint | Input | Output |
|---|---|---|
| `GET /api/health` | — | `{"status","version","device","models"}` |
| `POST /api/classify` | `multipart file`, `top_k` (1–5) | `{"predictions":[{"label","score"}],"confident"}` + header `X-Process-Time-ms` |
| `POST /api/detect` | `multipart file`, `conf` (0.05–0.95) | `{"detections":[{"label","score","box_xyxy"}],"summary","image":"data:image/jpeg;base64,…"}` |
| `POST /api/search/text` | `{"query","k"}` (k 1–24) | `{"results":[{"id","score","label","url"}]}` |
| `POST /api/search/image` | `multipart file`, `k` | như trên |
| `GET /api/gallery/{id}` | id | file ảnh tĩnh |
| `POST /api/chat` | `{"question"/"message","history","model","base_url","api_key"}` | SSE `text/event-stream`: `sources` → `token`* → `done` |
| `POST /api/chat/sync` | như trên | `{"answer","sources"}` (non-streaming) |
| `GET /api/chat/models` | — | `{"models":[…]}` |

Thiếu model → `503`, file sai → `400`, body sai → `422`.

## Docker
```bash
docker build -t ai-web-apps . && docker run -p 7860:7860 ai-web-apps   # mở http://localhost:7860
```

Chỉ số mô hình: xem `MODEL_CARD.md` và `artifacts/*/metrics.json` —
classifier test-acc 0.948 / macro-F1 0.947, detector mAP50 0.6707 / mAP50-95 0.5034 (COCO128),
RAG Hit@1 0.9286 / Hit@3 1.0 (14 câu hỏi mẫu).
