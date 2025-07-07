import json
import os
import uuid

# === CONFIG ===
INPUT_PATH = "no_api/chunking/length/len1.json"
OUTPUT_PATH = "no_api/results/len/vector_metadata1.json"

# Tạo thư mục nếu chưa tồn tại
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# === Load input ===
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    raw_docs = json.load(f)  # Dạng list

# === Convert format ===
metadata_list = []

for doc in raw_docs:
    url = doc.get("url", "")
    chunks = doc.get("chunks", [])

    for i, chunk in enumerate(chunks):
        content = chunk.get("content", "").strip()
        if content:
            unique_id = str(uuid.uuid4())
            metadata_list.append(
                {"id": unique_id, "text": content, "url": url, "chunk_index": i}
            )

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata_list, f, indent=2, ensure_ascii=False)

print(f" Đã tạo {len(metadata_list)} metadata items và lưu vào {OUTPUT_PATH}")
