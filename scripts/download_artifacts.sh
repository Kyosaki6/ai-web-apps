#!/usr/bin/env bash
# Tải artifacts nặng (không push lên git) về đúng chỗ.
# Điền link Google Drive / Hugging Face Release của nhóm vào các biến dưới rồi chạy:
#   bash scripts/download_artifacts.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CLASSIFIER_URL="${CLASSIFIER_URL:-TODO_LINK_model_pt}"   # ~45MB model.pt
YOLO_URL="${YOLO_URL:-TODO_LINK_yolo11n_pt}"
RETRIEVAL_URL="${RETRIEVAL_URL:-TODO_LINK_index_faiss_tar_gz}"

mkdir -p "$ROOT/artifacts/classifier" "$ROOT/artifacts/detector" "$ROOT/artifacts/retrieval"

if [[ "$CLASSIFIER_URL" == TODO* || "$YOLO_URL" == TODO* ]]; then
  echo "[!] Chua co link artifacts. Up file len Drive/HF Release roi dien vao script nay."
  echo "    Can: artifacts/classifier/model.pt (+classes.json, metrics.json, split.json)"
  echo "         artifacts/detector/yolo11n.pt (+metrics.json)"
  echo "         artifacts/retrieval/index.faiss (+meta.json, metrics.json)"
  exit 1
fi

curl -L -o "$ROOT/artifacts/classifier/model.pt" "$CLASSIFIER_URL"
curl -L -o "$ROOT/artifacts/detector/yolo11n.pt" "$YOLO_URL"
curl -L -o /tmp/retrieval.tar.gz "$RETRIEVAL_URL"
tar -xzf /tmp/retrieval.tar.gz -C "$ROOT/artifacts/retrieval"
echo "[OK] Artifacts da tai ve."
