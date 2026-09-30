"""AI-03: dựng gallery (COCO128 + Flowers), index CLIP+FAISS và metrics.json.

Chạy từ repo root (cần torch, transformers>=4.56, faiss-cpu):
    python scripts/build_gallery.py                 # mặc định 100 ảnh/loài hoa
    python scripts/build_gallery.py --per-class 20 --zip

Sinh ra:
    data/gallery/{coco,flowers}/...                 ảnh đã thu nhỏ (JPEG)
    artifacts/retrieval/{index.faiss, meta.json, metrics.json}
    artifacts/gallery.zip                           (nếu --zip) để upload Drive/HF Release
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from config import ART_DIR, CLIP_MODEL, ROOT, resolve_path
from core.retrieval import ClipEncoder, ImageSearch, build_index

EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SPLIT_NAMES = {"train", "val", "valid", "test", "images"}
UNLABELED = "unlabeled"


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in EXTS)


def save_small(src: Path, dst: Path, size: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size))
        im.save(dst, "JPEG", quality=85)


def rel(p: Path) -> str:
    """Đường dẫn TƯƠNG ĐỐI so với ROOT (bắt buộc cho meta.json)."""
    return p.resolve().relative_to(ROOT.resolve()).as_posix()


def coco_items(coco_dir: Path, gallery: Path, size: int, conf: float) -> list[dict]:
    from core.detector import ObjectDetector  # của Tuấn; nhãn = vật thể có điểm cao nhất

    det = ObjectDetector()
    items = []
    for p in list_images(coco_dir):
        with Image.open(p) as im:
            img = im.convert("RGB")
        result, _ = det.detect(img, conf=conf)
        dets = result.get("detections", [])
        label = max(dets, key=lambda d: d["score"])["label"] if dets else UNLABELED
        dst = gallery / "coco" / f"{p.stem}.jpg"
        save_small(p, dst, size)
        items.append({"path": rel(dst), "label": label, "source": "coco128"})
    print(f"COCO128: {len(items)} ảnh")
    return items


def flower_items(flowers_dir: Path, gallery: Path, size: int, per_class: int, seed: int) -> list[dict]:
    groups: dict[str, list[Path]] = {}
    for p in list_images(flowers_dir):
        groups.setdefault(p.parent.name, []).append(p)
    if any(g.lower() in SPLIT_NAMES for g in groups):
        print("CẢNH BÁO: có thư mục kiểu train/val/test lẫn với thư mục loài; nhãn có thể sai. "
              "Cần cấu trúc data/flowers/<tên_loài>/*.jpg", file=sys.stderr)
    rng = random.Random(seed)
    items = []
    for cls, paths in sorted(groups.items()):
        rng.shuffle(paths)
        for j, p in enumerate(paths[:per_class]):
            dst = gallery / "flowers" / cls / f"{j:03d}.jpg"
            save_small(p, dst, size)
            items.append({"path": rel(dst), "label": cls.replace("_", " "), "source": "flowers"})
    print(f"Flowers: {len(items)} ảnh, {len(groups)} loài")
    return items


def evaluate(search: ImageSearch, items: list[dict], n_queries: int, seed: int) -> dict:
    """P@5 ảnh→ảnh (bỏ chính nó) và P@10 chữ→ảnh; 'đúng' = cùng nhãn."""
    embs = search.index.reconstruct_n(0, search.index.ntotal)
    labels = np.array([it["label"] for it in items])
    labeled = np.where(labels != UNLABELED)[0]

    q = np.random.default_rng(seed).choice(labeled, size=min(n_queries, len(labeled)), replace=False)
    sims = embs[q] @ embs.T
    sims[np.arange(len(q)), q] = -np.inf
    top5 = np.argsort(-sims, axis=1)[:, :5]
    p5 = float((labels[top5] == labels[q][:, None]).mean())

    uniq = sorted(set(labels[labeled]))
    text_vecs = search.encoder.encode_texts([f"a photo of a {name}" for name in uniq])
    _, ids = search.index.search(text_vecs, 10)
    p10 = float(np.mean([(labels[ids[i]] == name).mean() for i, name in enumerate(uniq)]))

    return {
        "image_to_image_P@5": round(p5, 4),
        "text_to_image_P@10": round(p10, 4),
        "n_images": len(items),
        "n_labels": len(uniq),
        "n_image_queries": int(len(q)),
        "clip_model": CLIP_MODEL,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-dir", default="data/coco128")
    ap.add_argument("--flowers-dir", default="data/flowers")
    ap.add_argument("--gallery", default="data/gallery")
    ap.add_argument("--per-class", type=int, default=100, help="số ảnh mỗi loài hoa")
    ap.add_argument("--size", type=int, default=256, help="cạnh dài tối đa khi lưu gallery")
    ap.add_argument("--conf", type=float, default=0.25, help="ngưỡng detector cho nhãn COCO")
    ap.add_argument("--queries", type=int, default=1000, help="số ảnh truy vấn để tính P@5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-coco", action="store_true", help="bỏ COCO128 (khi chưa có weights YOLO)")
    ap.add_argument("--zip", action="store_true", help="nén data/gallery thành artifacts/gallery.zip")
    args = ap.parse_args()

    gallery = resolve_path(args.gallery)
    out_dir = ART_DIR / "retrieval"

    items: list[dict] = []
    if not args.no_coco:
        items += coco_items(resolve_path(args.coco_dir), gallery, args.size, args.conf)
    items += flower_items(resolve_path(args.flowers_dir), gallery, args.size, args.per_class, args.seed)
    if not items:
        sys.exit("Không tìm thấy ảnh nào. Kiểm tra --coco-dir / --flowers-dir.")

    encoder = ClipEncoder()
    build_index(encoder, items, out_dir)
    print(f"Đã lưu index ({len(items)} ảnh) vào {out_dir}")

    metrics = evaluate(ImageSearch(out_dir, encoder=encoder), items, args.queries, args.seed)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.zip:
        # zip chứa data/gallery/... nên giải nén tại repo root là đúng đường dẫn trong meta.json
        base = ART_DIR / "gallery"
        shutil.make_archive(str(base), "zip", root_dir=ROOT, base_dir=gallery.relative_to(ROOT).as_posix())
        print(f"Đã nén: {base}.zip")


if __name__ == "__main__":
    main()
