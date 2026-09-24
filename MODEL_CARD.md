# MODEL CARD — AI Web Apps

## 1. Phân loại hoa (classifier)
- **Dữ liệu:** TF Flowers, 3670 ảnh, 5 lớp (`daisy`, `dandelion`, `roses`, `sunflowers`, `tulips`), chia 80/10/10 (`artifacts/classifier/split.json`).
- **Mô hình:** ResNet-18 fine-tune, 5 epochs (giữ nguyên `TRAIN_TF`/`EVAL_TF` trong `core/classifier.py`).
- **Chỉ số (đo trên Colab T4):**
  - Test accuracy: _TODO — dán từ `artifacts/classifier/metrics.json` (target > 88%)_
  - F1 macro: _TODO_
- **Giới hạn:** ảnh ngoài 5 loài vẫn bị gán một trong 5 nhãn → client phải kiểm tra cờ `confident` trong `POST /api/classify`.

## 2. Phát hiện đối tượng (detector)
- **Mô hình:** YOLO11n, pretrain COCO (`artifacts/detector/yolo11n.pt`).
- **Chỉ số:** mAP50 / mAP50-95 trên COCO128 — _TODO (Tuấn dán từ `model.val`)_.

## 3. Tìm kiếm ảnh (retrieval)
- **Mô hình:** CLIP ViT-B/32 + FAISS (`artifacts/retrieval/index.faiss`).
- **Chỉ số:** image→image P@5, text→image P@10 — _TODO (Khánh dán từ `metrics.json`)_.

## 4. Chatbot RAG (llm)
- **Pipeline:** chunk 600 ký tự (`data/kb/*.md`) → `paraphrase-multilingual-MiniLM-L12-v2` → Qwen2.5-1.5B-Instruct (GPU) / 0.5B (CPU).
- **Chỉ số:** Hit@1 / Hit@3 trên ~10 câu hỏi mẫu — _TODO (Thái, target Hit@3 = 100%)_.
- **An toàn:** system prompt bắt citation `[file.md]`, câu fallback khi thiếu thông tin, coi TÀI LIỆU là dữ liệu (chống prompt-injection).
