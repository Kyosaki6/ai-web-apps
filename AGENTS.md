# AGENTS.md — ai-web-apps

Single FastAPI backend (`api/main.py`) serving 4 models; two thin frontends call it: Streamlit (`streamlit_app.py`) and React (`web/`, Vite).

## Commands (Python 3.11, Node 22)

```bash
pip install -r requirements.txt                       # backend; CPU: install torch CPU build first (see Dockerfile)
uvicorn api.main:app --port 8000                      # backend; also serves web/dist if built
API_URL=http://localhost:8000 streamlit run streamlit_app.py  # Streamlit is API-only client
cd web && npm install && npm run dev                  # React dev, proxies /api → :8000
ENABLED_MODELS= pytest -q                             # tests, no GPU/models needed (fakes injected)
docker build -t ai-web-apps . && docker run -p 7860:7860 ai-web-apps
```

CI (`.github/workflows/pytest.yml`): installs only `fastapi httpx pytest pillow pydantic python-multipart`, runs `pytest -q`. No lint/typecheck configured.

## Architecture

- `config.py` is single source of truth — every value overridable by env (`ENABLED_MODELS`, `LLM_MODEL`, `CLIP_MODEL`, `EMBED_MODEL`, `YOLO_WEIGHTS`, `MAX_UPLOAD_MB`, `CORS_ORIGINS`, `PORT`, `API_URL`). Change model/env behavior there, not in callers. `resolve_path()` keeps relative paths portable (Docker/HF Spaces).
- `api/main.py`: lazy `LOADERS` dict (`classifier|detector|retrieval|llm` → `core/*`) loads only names in `ENABLED_MODELS`; one loader failing logs and continues. Missing model → `503` via `_require()`. Conventions: clamp form params (`top_k` 1–5, `conf` 0.05–0.95, `k` 1–24), images returned as `data:image/jpeg;base64,` strings, every response adds `X-Process-Time-ms`.
- `core/{classifier,detector,retrieval,llm}.py`: one class per model. Classifier must return `confident` flag (out-of-scope images still get a label — clients check it).
- `/api/chat` is SSE-over-POST (`sources` → `token`* → `done`); `EventSource` can't POST so both clients parse the stream manually (`web/src/api.js:streamChat`, `streamlit_app.py`). Keep `/api/chat/sync` for non-streaming.
- `web/src/api.js` owns all fetch calls; `API_BASE` empty = same origin when FastAPI serves `web/dist` (built via Docker stage 1; `web/dist/` is gitignored). Dev proxy is in `web/vite.config.js`.
- `conftest.py` inserts repo root into `sys.path` and stubs `torch` if uninstalled — tests must stay torch-free.

## Data / artifacts (gitignored, never commit)

- Heavy files excluded: `artifacts/**/*.pt`, `artifacts/**/index.faiss`, `data/gallery/*`, `data/flowers/`, `data/coco128/`, `web/dist/`, `.env`. Fetch weights via `bash scripts/download_artifacts.sh` (URLs currently TODO placeholders). Metrics live in `artifacts/*/metrics.json`.
- RAG corpus: `data/kb/*.md`, chunk 600 chars; chunking/embedding assumptions documented in `MODEL_CARD.md`.
