import json
import uuid

# === CONFIG ===
INPUT_PATH = "no_api/chunking/sem_len/sem_len.json"
OUTPUT_PATH = "no_api/results/sem_len/vector_metadata.json"

# === Load input (JSON lines) ===
raw_docs = []
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            raw_docs.append(json.loads(line))

# === Convert format ===
metadata_list = []
for doc in raw_docs:
    url = doc.get("Url", "")
    chunks = doc.get("Chunks", [])

    for i, chunk in enumerate(chunks):
        content = chunk.get("content", "").strip()
        if content:
            unique_id = str(uuid.uuid4())  # tạo id cố định
            metadata_list.append(
                {
                    "id": unique_id,
                    "text": content,
                    "url": url,
                    "chunk_index": i,
                }
            )

# === Save to file ===
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata_list, f, indent=2, ensure_ascii=False)

print(f" Đã tạo {len(metadata_list)} metadata items và lưu vào {OUTPUT_PATH}")
