#!/usr/bin/env python3
import argparse
import http.cookiejar
import os
import re
import sys
import tarfile
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "artifacts"

GDRIVE_ROOT_FOLDER_ID = "1dHuzQ9R9qtHtDkAs2H3VKWg9yUBMziC9"
GDRIVE_ROOT_URL = f"https://drive.google.com/drive/folders/{GDRIVE_ROOT_FOLDER_ID}?usp=sharing"

FALLBACK_ARTIFACTS = {
    "classifier/model.pt": "1lsNCptZZJT7Mn8FqYpJ5K48ideRKWnQQ",
    "classifier/classes.json": "1_8ZNPIMTNhtJlRChWlQoLwAAMxN8Y1U6",
    "classifier/metrics.json": "13dMbnBML58x0fWBNU-aoFVIR1EVv7fhY",
    "classifier/split.json": "1s59pUvr_aFpIU0yOI4k7U0tTt0UIfXUz",
    "detector/yolo11n.pt": "1xEXJpgqe3gpcNTqX7oOUsJoOuukAVRAD",
    "detector/metrics.json": "11andoRtSvWQkF8rGbSwphxbhafDhXyXj",
    "retrieval/index.faiss": "1HPfSai-YWL_W9-70rUlUPQZdyTsMenw1",
    "retrieval/meta.json": "1zpRTsLELQwGyV2E0d49gQWYz__pDU5BW",
    "retrieval/metrics.json": "1Cd0MTSonbupumYAnmVpK7_7hNF8oNIVd",
    "rag/metrics.json": "1lxNfn-YYK8iut9jQkOnTVQhkBNy9zOjR",
    "benchmark_metrics.json": "1MwroYCDSCtD0CdlZe1_camiIW8Zjl0lU",
    "rag_metrics.json": "1FxhVWhSKjOPr5DMNgPadfOxCagPA4Hf6",
}

FLOWERS_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
COCO128_URL = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"


class Downloader:
    def __init__(self):
        self.cj = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.cj))

    def _open(self, url: str):
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
            },
        )
        return self.opener.open(req)

    def download_file(self, url: str, dest_path: Path, label: str = "") -> None:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

        resp = self._open(url)
        content_type = resp.headers.get("Content-Type", "")

        if "text/html" in content_type:
            html = resp.read().decode("utf-8", errors="ignore")
            inputs = dict(re.findall(r'<input[^>]*name="([^"]*)"[^>]*value="([^"]*)"', html))
            if "confirm" in inputs:
                query = urllib.parse.urlencode(inputs)
                confirm_url = f"https://drive.usercontent.google.com/download?{query}"
                resp = self._open(confirm_url)

        total_bytes = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024
        start_time = time.time()

        tag = f"[{label}]" if label else ""
        size_str = f" ({total_bytes / (1024 * 1024):.1f} MB)" if total_bytes > 0 else ""
        print(f"{tag} Tải về: {dest_path.name}{size_str}...")

        with open(temp_path, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                elapsed = time.time() - start_time
                speed = downloaded / (elapsed if elapsed > 0 else 1) / (1024 * 1024)

                if total_bytes > 0:
                    pct = (downloaded / total_bytes) * 100
                    sys.stdout.write(
                        f"\r  -> {pct:5.1f}% [{downloaded / (1024*1024):.1f}/{total_bytes / (1024*1024):.1f} MB] "
                        f"@ {speed:.2f} MB/s"
                    )
                else:
                    sys.stdout.write(
                        f"\r  -> {downloaded / (1024*1024):.1f} MB @ {speed:.2f} MB/s"
                    )
                sys.stdout.flush()

        print("\n  -> Hoàn thành.")
        os.replace(temp_path, dest_path)

    def download_gdrive_file(self, file_id: str, dest_path: Path, label: str = "") -> None:
        url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download"
        self.download_file(url, dest_path, label=label)

    def scrape_folder_contents(self, folder_id: str) -> tuple[dict[str, str], dict[str, str]]:
        url = f"https://drive.google.com/drive/folders/{folder_id}?usp=sharing"
        try:
            resp = self._open(url)
            raw = resp.read().decode("latin1", errors="ignore")
            decoded = re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), raw)
            pattern = r'\["([a-zA-Z0-9_-]{25,})",\["' + re.escape(folder_id) + r'"\],"([^"]+)","([^"]+)"'
            matches = re.findall(pattern, decoded)
            files = {}
            subfolders = {}
            for cid, name, mime in matches:
                if "folder" in mime:
                    subfolders[name] = cid
                else:
                    if name != ".gitkeep":
                        files[name] = cid
            return files, subfolders
        except Exception:
            return {}, {}

    def discover_all_artifacts(self, root_folder_id: str) -> dict[str, str]:
        root_files, subfolders = self.scrape_folder_contents(root_folder_id)
        if not root_files and not subfolders:
            return FALLBACK_ARTIFACTS

        result = {}
        for fname, fid in root_files.items():
            result[fname] = fid

        for sub_name, sub_id in subfolders.items():
            sub_files, _ = self.scrape_folder_contents(sub_id)
            for fname, fid in sub_files.items():
                rel_path = f"{sub_name}/{fname}"
                result[rel_path] = fid

        for rel_path, fid in FALLBACK_ARTIFACTS.items():
            if rel_path not in result:
                result[rel_path] = fid

        return result


def extract_folder_id(url_or_id: str) -> str:
    match = re.search(r"folders/([a-zA-Z0-9_-]{25,})", url_or_id)
    if match:
        return match.group(1)
    return url_or_id


def download_artifacts(dl: Downloader, folder_input: str, force: bool = False) -> None:
    print("=== 1. TẢI CÁC MÔ HÌNH VÀ ARTIFACTS ===")
    folder_id = extract_folder_id(folder_input)
    artifacts = dl.discover_all_artifacts(folder_id)

    for rel_path, file_id in sorted(artifacts.items()):
        dest = ART_DIR / rel_path
        if dest.exists() and not force:
            print(f"[OK] {rel_path} đã tồn tại, bỏ qua.")
            continue
        dl.download_gdrive_file(file_id, dest, label=rel_path)


def download_flowers(dl: Downloader, force: bool = False) -> None:
    print("\n=== 2. TẢI DATASET FLOWERS (TF Flowers) ===")
    flowers_dir = ROOT / "data" / "flowers"
    sample_dir = flowers_dir / "flower_photos"

    if sample_dir.exists() and any(sample_dir.glob("*/*.jpg")) and not force:
        count = len(list(sample_dir.glob("*/*.jpg")))
        print(f"[OK] Dataset flowers đã tồn tại ({count} ảnh) tại {sample_dir.relative_to(ROOT)}. Bỏ qua.")
        return

    tar_path = flowers_dir / "flower_photos.tgz"
    flowers_dir.mkdir(parents=True, exist_ok=True)
    dl.download_file(FLOWERS_URL, tar_path, label="TF Flowers Dataset")

    print("  -> Đang giải nén flower_photos.tgz...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=flowers_dir)

    if tar_path.exists():
        tar_path.unlink()
    print(f"[OK] Đã giải nén xong vào {sample_dir.relative_to(ROOT)}.")


def download_coco128(dl: Downloader, force: bool = False) -> None:
    print("\n=== 3. TẢI DATASET COCO128 ===")
    coco_dir = ROOT / "data" / "coco128"
    images_dir = coco_dir / "images"

    if images_dir.exists() and any(images_dir.glob("**/*.jpg")) and not force:
        count = len(list(images_dir.glob("**/*.jpg")))
        print(f"[OK] Dataset coco128 đã tồn tại ({count} ảnh) tại {coco_dir.relative_to(ROOT)}. Bỏ qua.")
        return

    zip_path = ROOT / "data" / "coco128.zip"
    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    dl.download_file(COCO128_URL, zip_path, label="COCO128 Dataset")

    print("  -> Đang giải nén coco128.zip...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(path=ROOT / "data")

    if zip_path.exists():
        zip_path.unlink()
    print(f"[OK] Đã giải nén xong vào {coco_dir.relative_to(ROOT)}.")


def main():
    parser = argparse.ArgumentParser(description="Tải models và datasets cho ai-web-apps.")
    parser.add_argument("--models", "--artifacts", dest="models", action="store_true", help="Chỉ tải các file mô hình và artifacts")
    parser.add_argument("--flowers", action="store_true", help="Chỉ tải dataset TF Flowers")
    parser.add_argument("--coco128", action="store_true", help="Chỉ tải dataset COCO128")
    parser.add_argument("--folder-url", default=GDRIVE_ROOT_URL, help="URL Google Drive folder chứa artifacts")
    parser.add_argument("--force", action="store_true", help="Ghi đè file nếu đã có")
    args = parser.parse_args()

    download_all = not (args.models or args.flowers or args.coco128)
    dl = Downloader()

    if download_all or args.models:
        download_artifacts(dl, args.folder_url, force=args.force)

    if download_all or args.flowers:
        download_flowers(dl, force=args.force)

    if download_all or args.coco128:
        download_coco128(dl, force=args.force)

    print("\n[THÀNH CÔNG] Tất cả dữ liệu và models đã sẵn sàng.")


if __name__ == "__main__":
    main()
