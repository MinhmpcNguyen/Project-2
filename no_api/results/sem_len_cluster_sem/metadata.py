import json
import os
import uuid

# === CONFIG ===
INPUT_PATH = "no_api/chunking/cluster/sem_len_cluster_rechunked.json"
OUTPUT_PATH = "no_api/results/sem_len_cluster_sem/vector_metadata.json"

# Tạo thư mục nếu chưa tồn tại
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# === Load input ===
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    cluster_data = json.load(f)  # Dict: cluster_0, cluster_1, ...

# === Convert format ===
metadata_list = []

for cluster_key, cluster_info in cluster_data.items():
    chunks = cluster_info.get("Chunks", [])

    for i, chunk in enumerate(chunks):
        content = chunk.get("content", "").strip()
        urls = chunk.get("source_urls", [])
        if not content:
            continue

        # Tạo ID duy nhất gắn với nội dung cụ thể để đồng bộ hóa giữa các bước
        uid = str(uuid.uuid4())

        metadata_list.append(
            {
                "id": uid,
                "text": content,
                "url": urls,  # List các URL nguồn
                "chunk_index": i,
                "cluster": cluster_key,  # Ghi lại cluster nếu cần truy ngược
            }
        )

# === Save to file ===
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata_list, f, indent=2, ensure_ascii=False)

print(
    f" Đã tạo {len(metadata_list)} metadata items từ các cluster và lưu vào {OUTPUT_PATH}"
)
