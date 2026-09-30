"""OPS-02: Benchmark hiệu năng hệ thống (đo p50, p95, throughput, RAM, CPU).

Chạy trực tiếp (test in-process không cần bật uvicorn trước):
    python scripts/benchmark.py

Hoặc benchmark server đang chạy:
    python scripts/benchmark.py --url http://localhost:8000
"""
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import psutil
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from config import ART_DIR, ROOT, resolve_path


def create_sample_image() -> bytes:
    # Thử lấy ảnh thật từ dataset hoa nếu có, nếu không tạo ảnh mẫu
    flowers_sample = list(Path("data/flowers").rglob("*.jpg"))
    if flowers_sample:
        return flowers_sample[0].read_bytes()
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color=(255, 128, 0)).save(buf, format="JPEG")
    return buf.getvalue()


def run_benchmark(client, sample_img: bytes, n_requests: int = 50) -> dict:
    process = psutil.Process(os.getpid())
    results = {}

    benchmarks = [
        {
            "name": "GET /api/health",
            "call": lambda: client.get("/api/health"),
            "n": n_requests,
        },
        {
            "name": "POST /api/classify",
            "call": lambda: client.post("/api/classify", files={"file": ("img.jpg", sample_img, "image/jpeg")}),
            "n": min(20, n_requests),
        },
        {
            "name": "POST /api/detect",
            "call": lambda: client.post("/api/detect", files={"file": ("img.jpg", sample_img, "image/jpeg")}),
            "n": min(15, n_requests),
        },
        {
            "name": "POST /api/search/text",
            "call": lambda: client.post("/api/search/text", json={"query": "yellow flower", "k": 4}),
            "n": min(15, n_requests),
        },
    ]

    print(f"{'Endpoint':<25} | {'Count':<6} | {'p50 (ms)':<10} | {'p95 (ms)':<10} | {'Status'}")
    print("-" * 70)

    for bench in benchmarks:
        name = bench["name"]
        n = bench["n"]
        latencies = []
        statuses = []

        # warm up 1 call
        try:
            r = bench["call"]()
            statuses.append(r.status_code)
        except Exception as e:
            statuses.append(str(e))

        t_start = time.perf_counter()
        for _ in range(n):
            t0 = time.perf_counter()
            r = bench["call"]()
            latencies.append((time.perf_counter() - t0) * 1000.0)
            statuses.append(r.status_code)
        total_time = time.perf_counter() - t_start

        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        throughput = round(n / total_time, 2)
        success_rate = round(statuses.count(200) / len(statuses) * 100, 1)

        results[name] = {
            "requests": n,
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "throughput_req_per_s": throughput,
            "success_rate_pct": success_rate,
        }

        print(f"{name:<25} | {n:<6} | {p50:<10.2f} | {p95:<10.2f} | {success_rate}% OK")

    mem_mb = process.memory_info().rss / 1024 / 1024
    cpu_pct = psutil.cpu_percent(interval=0.1)

    summary = {
        "benchmarks": results,
        "system": {
            "ram_rss_mb": round(mem_mb, 1),
            "cpu_percent": cpu_pct,
            "logical_cpus": psutil.cpu_count(),
        },
    }

    out_file = ART_DIR / "benchmark_metrics.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Đã lưu kết quả benchmark vào: {out_file}")
    print(f"RAM sử dụng: {mem_mb:.1f} MB | CPU: {cpu_pct}%")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="Số lượt request mỗi endpoint")
    args = parser.parse_args()

    sample_img = create_sample_image()
    print("Khởi động FastAPI lifespan và nạp mô hình...")
    with TestClient(app) as client:
        run_benchmark(client, sample_img, n_requests=args.n)


if __name__ == "__main__":
    main()
