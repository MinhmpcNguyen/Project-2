import json


def chunk_by_length(content_list, max_words=1024, min_words=30):
    chunks = []
    current_chunk = []
    current_len = 0

    for para in content_list:
        para = para.strip()
        word_count = len(para.split())

        if word_count == 0:
            continue

        if current_len + word_count <= max_words:
            current_chunk.append(para)
            current_len += word_count
        else:
            if current_len >= min_words:
                chunks.append(" ".join(current_chunk))
                current_chunk = [para]
                current_len = word_count
            else:
                current_chunk.append(para)
                current_len += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


input_file = "no_api/reformat/processed_results/final_output.json"
output_file = "no_api/chunking/length/len1.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

output = []
for doc in data:
    chunks = chunk_by_length(doc["content"], max_words=350, min_words=30)
    output.append(
        {"url": doc["url"], "chunks": [{"content": chunk} for chunk in chunks]}
    )

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print(f" Chunked data saved to {output_file}")
