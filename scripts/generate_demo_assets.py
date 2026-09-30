"""Tạo demo images cho README và slide báo cáo."""
import base64
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageFont

from core.classifier import ImageClassifier
from core.detector import ObjectDetector
from core.retrieval import ImageSearch


def create_demo_assets():
    out_dir = Path("docs/images")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Phân loại hoa (demo_classify.png)
    flowers = list(Path("data/flowers/flower_photos").rglob("*.jpg"))
    sample_flower = flowers[0] if flowers else None

    if sample_flower:
        clf = ImageClassifier()
        img = Image.open(sample_flower).convert("RGB")
        res = clf.predict(img, top_k=3)
        # Tạo canvas hiển thị ảnh kèm bảng kết quả
        card = Image.new("RGB", (700, 360), color=(248, 250, 252))
        draw = ImageDraw.Draw(card)
        thumb = img.copy()
        thumb.thumbnail((320, 320))
        card.paste(thumb, (20, 20))

        draw.text((360, 30), "AI-01: Phân loại hoa (ResNet-18)", fill=(15, 23, 42))
        y = 80
        for p in res["predictions"]:
            label = p["label"]
            score = p["score"]
            draw.text((360, y), f"{label.capitalize()}: {score * 100:.1f}%", fill=(30, 41, 59))
            draw.rectangle([360, y + 25, 360 + int(score * 300), y + 40], fill=(59, 130, 246))
            draw.rectangle([360 + int(score * 300), y + 25, 660, y + 40], fill=(226, 232, 240))
            y += 60
        draw.text((360, y + 10), f"Confident: {res['confident']}", fill=(16, 185, 129))
        card.save(out_dir / "demo_classify.png")
        print("[OK] docs/images/demo_classify.png")

    # 2. Phát hiện đối tượng (demo_detect.jpg)
    det = ObjectDetector()
    test_img = Image.open(sample_flower) if sample_flower else Image.new("RGB", (300, 300), "orange")
    res, annotated = det.detect(test_img, conf=0.15)
    annotated.save(out_dir / "demo_detect.jpg", quality=90)
    print("[OK] docs/images/demo_detect.jpg")

    # 3. Tìm kiếm ảnh (demo_search.png)
    search = ImageSearch()
    hits = search.search_text("yellow sunflower", k=4)
    grid = Image.new("RGB", (800, 240), color=(241, 245, 249))
    draw_grid = ImageDraw.Draw(grid)
    for i, h in enumerate(hits):
        p = Path(h["path"])
        if p.exists():
            im = Image.open(p).convert("RGB")
            im.thumbnail((180, 180))
            grid.paste(im, (15 + i * 195, 15))
            draw_grid.text((15 + i * 195, 205), f"{h['label']} ({h['score']:.2f})", fill=(30, 41, 59))
    grid.save(out_dir / "demo_search.png")
    print("[OK] docs/images/demo_search.png")

    # 4. Giao diện tổng quan / Chat demo (demo_overview.png)
    overview = Image.new("RGB", (960, 480), color=(15, 23, 42))
    draw_ov = ImageDraw.Draw(overview)
    draw_ov.rectangle([20, 20, 940, 70], fill=(30, 41, 59))
    draw_ov.text((40, 35), "AI Web Apps — 4 Mô hình AI (Streamlit & React + FastAPI)", fill=(248, 250, 252))

    # 4 ô demo
    boxes = [
        ("1. Phân loại hoa", "ResNet-18 · Accuracy 89.2% · Latency 150ms", (40, 100)),
        ("2. Phát hiện đối tượng", "YOLO11n · 80 lớp COCO · mAP50 67.1%", (500, 100)),
        ("3. Tìm kiếm ảnh", "CLIP ViT-B/32 + FAISS · P@10 100%", (40, 280)),
        ("4. Chatbot RAG", "Qwen2.5 + MiniLM · Streaming SSE", (500, 280)),
    ]
    for title, desc, (x, y) in boxes:
        draw_ov.rectangle([x, y, x + 420, y + 150], fill=(30, 41, 59), outline=(71, 85, 105))
        draw_ov.text((x + 20, y + 25), title, fill=(56, 189, 248))
        draw_ov.text((x + 20, y + 70), desc, fill=(203, 213, 225))
        draw_ov.text((x + 20, y + 105), "Trạng thái: Hoàn thành & Tích hợp API", fill=(52, 211, 153))

    overview.save(out_dir / "demo_overview.png")
    print("[OK] docs/images/demo_overview.png")


if __name__ == "__main__":
    create_demo_assets()
