"""Đánh giá độ chính xác của bộ truy xuất RAG (Retriever).
Đo Hit@1, Hit@3 và thời gian truy xuất trung bình (latency_ms).
Kết quả ghi vào artifacts/rag/metrics.json và artifacts/rag_metrics.json.
"""
import json
import os
import sys
import time
from pathlib import Path

# Đảm bảo import được config và core
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.llm import Retriever, load_chunks

# Tập câu hỏi chuẩn kiểm tra toàn diện 6 tài liệu chính sách ShopLite
TEST_CASES = [
    {
        "query": "Đổi trả trong mấy ngày?",
        "expected_source": "doi_tra.md",
        "keywords": ["7 ngày", "đổi trả"],
    },
    {
        "query": "Thời trang và giày dép được đổi size trong bao nhiêu ngày?",
        "expected_source": "doi_tra.md",
        "keywords": ["14 ngày", "đổi size"],
    },
    {
        "query": "Những sản phẩm nào không áp dụng đổi trả?",
        "expected_source": "doi_tra.md",
        "keywords": ["đồ lót", "mỹ phẩm đã mở nắp", "thực phẩm"],
    },
    {
        "query": "Giao hàng nội thành Hà Nội và TP HCM mất bao lâu?",
        "expected_source": "giao_hang.md",
        "keywords": ["1–2 ngày", "1-2 ngày", "Hà Nội"],
    },
    {
        "query": "Đơn hàng từ bao nhiêu tiền thì được freeship miễn phí giao hàng?",
        "expected_source": "giao_hang.md",
        "keywords": ["300.000đ", "miễn phí"],
    },
    {
        "query": "Phí giao hàng hỏa tốc trong 2 giờ là bao nhiêu?",
        "expected_source": "giao_hang.md",
        "keywords": ["50.000đ", "hỏa tốc"],
    },
    {
        "query": "Thanh toán khi nhận hàng COD tối đa bao nhiêu tiền?",
        "expected_source": "thanh_toan.md",
        "keywords": ["10.000.000đ", "COD"],
    },
    {
        "query": "Điều kiện để trả góp 0% lãi suất là gì?",
        "expected_source": "thanh_toan.md",
        "keywords": ["3.000.000đ", "trả góp 0%"],
    },
    {
        "query": "Thiết bị điện tử được bảo hành mấy tháng?",
        "expected_source": "bao_hanh.md",
        "keywords": ["12 tháng", "thiết bị điện tử"],
    },
    {
        "query": "Thời gian xử lý bảo hành mất bao lâu?",
        "expected_source": "bao_hanh.md",
        "keywords": ["7 đến 15 ngày", "xử lý"],
    },
    {
        "query": "Chi tiêu bao nhiêu để được lên hạng thành viên Bạc?",
        "expected_source": "khach_hang_than_thiet.md",
        "keywords": ["5.000.000đ", "hạng Bạc"],
    },
    {
        "query": "Một điểm ShopLite quy đổi được bao nhiêu tiền?",
        "expected_source": "khach_hang_than_thiet.md",
        "keywords": ["1.000đ", "1 điểm"],
    },
    {
        "query": "Đăng nhập sai mấy lần thì bị khóa tài khoản?",
        "expected_source": "tai_khoan.md",
        "keywords": ["5 lần", "15 phút"],
    },
    {
        "query": "Link đặt lại mật khẩu có hiệu lực trong bao lâu?",
        "expected_source": "tai_khoan.md",
        "keywords": ["30 phút", "quên mật khẩu"],
    },
]


def evaluate():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from data/kb/")
    retriever = Retriever(chunks)

    hit1_count = 0
    hit3_count = 0
    total = len(TEST_CASES)
    latencies = []
    details = []

    for tc in TEST_CASES:
        t0 = time.perf_counter()
        results = retriever.search(tc["query"], k=3)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        sources = [r["source"] for r in results]
        h1 = sources[0] == tc["expected_source"] if sources else False
        h3 = tc["expected_source"] in sources[:3]

        if h1:
            hit1_count += 1
        if h3:
            hit3_count += 1

        details.append({
            "query": tc["query"],
            "expected": tc["expected_source"],
            "retrieved": sources,
            "hit@1": h1,
            "hit@3": h3,
            "latency_ms": round(elapsed_ms, 2),
            "top_score": results[0]["score"] if results else 0.0,
        })
        print(f"[{'OK' if h3 else 'FAIL'}] '{tc['query']}' -> top-3: {sources} (Hit@1={h1}, Hit@3={h3})")

    hit1_rate = round(hit1_count / total, 4)
    hit3_rate = round(hit3_count / total, 4)
    p50_latency = round(float(sorted(latencies)[len(latencies) // 2]), 2)
    avg_latency = round(float(sum(latencies) / len(latencies)), 2)

    metrics = {
        "model": retriever.embedder[0].auto_model.config._name_or_path if hasattr(retriever.embedder, "__getitem__") else "paraphrase-multilingual-MiniLM-L12-v2",
        "num_test_cases": total,
        "hit@1": hit1_rate,
        "hit@3": hit3_rate,
        "target_hit@3": 1.0,
        "hit@3_met": hit3_rate >= 1.0,
        "latency_ms_p50": p50_latency,
        "latency_ms_avg": avg_latency,
        "details": details,
    }

    out_dir = ROOT / "artifacts" / "rag"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Ghi thêm artifacts/rag_metrics.json để đồng bộ với mô tả README
    alt_file = ROOT / "artifacts" / "rag_metrics.json"
    with open(alt_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n--- KẾT QUẢ ĐÁNH GIÁ RAG RETRIEVER ---")
    print(f"Số câu hỏi đánh giá: {total}")
    print(f"Hit@1: {hit1_rate * 100:.1f}% ({hit1_count}/{total})")
    print(f"Hit@3: {hit3_rate * 100:.1f}% ({hit3_count}/{total})")
    print(f"Độ trễ trung bình: {avg_latency:.1f} ms (p50: {p50_latency:.1f} ms)")
    print(f"Target Hit@3 = 100%: {'ĐẠT' if hit3_rate >= 1.0 else 'CHƯA ĐẠT'}")
    print(f"Đã lưu kết quả tại: {out_file} và {alt_file}")

    return metrics


if __name__ == "__main__":
    evaluate()
