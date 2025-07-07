import json


def remove_duplicate_chunks_ndjson(input_path, output_path):
    seen_contents = set()

    with (
        open(input_path, "r", encoding="utf-8") as infile,
        open(output_path, "w", encoding="utf-8") as outfile,
    ):
        for line in infile:
            if not line.strip():
                continue  # Bỏ qua dòng trắng

            record = json.loads(line)
            unique_chunks = []

            for chunk in record.get("Chunks", []):
                content = chunk.get("content", "").strip()
                if content and content not in seen_contents:
                    seen_contents.add(content)
                    unique_chunks.append({"content": content})

            if unique_chunks:
                record["Chunks"] = unique_chunks
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done. Output saved to: {output_path}")


# Sử dụng:
remove_duplicate_chunks_ndjson(
    "no_api/chunking/sem_len/sem_len.json", "no_api/chunking/sem_len/output_dedup.json"
)
