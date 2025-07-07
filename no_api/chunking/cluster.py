import asyncio
import json
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

embedding_cache = {}
tokenizer = None
model = None


# Load HuggingFace transformer model
async def initialize_embedding_utils():
    global tokenizer, model
    model_name = "intfloat/e5-large-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return {"embedding": model_name}


# Embed một đoạn văn
async def create_embedding(paragraph: str):
    if paragraph in embedding_cache:
        return embedding_cache[paragraph]

    text = "query: " + paragraph.strip()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0]  # CLS token
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    result = embedding.cpu().numpy().squeeze()

    embedding_cache[paragraph] = result
    return result


# Gán đoạn văn vào các cụm (theo ngưỡng similarity và khoảng cách)
def clustering_paragraphs(
    embeddings: np.ndarray, similarity_threshold: float, distance_threshold: float
) -> Dict[int, List[int]]:
    clusters = {}
    cluster_centroids = []

    for idx, emb in enumerate(embeddings):
        assigned = False
        for cluster_id, centroid in enumerate(cluster_centroids):
            similarity = cosine_similarity([emb], [centroid])[0][0]
            distance = np.linalg.norm(emb - centroid)

            if similarity > similarity_threshold and distance < distance_threshold:
                clusters[cluster_id].append(idx)
                n = len(clusters[cluster_id])
                cluster_centroids[cluster_id] = centroid * (n - 1) / n + emb / n
                assigned = True
                break

        if not assigned:
            new_cluster_id = len(cluster_centroids)
            clusters[new_cluster_id] = [idx]
            cluster_centroids.append(emb)

    return clusters, cluster_centroids


# Đọc NDJSON và trả về list [{text, url}]
def load_paragraphs_with_source(file_path: str) -> List[Dict[str, str]]:
    paragraphs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            url = doc.get("Url", "")
            chunks = doc.get("Chunks", [])
            for chunk in chunks:
                if isinstance(chunk, dict) and "content" in chunk:
                    paragraphs.append({"text": chunk["content"].strip(), "url": url})
    return paragraphs


# Thực hiện cluster và gắn URL lại
async def cluster_paragraphs_only(
    paragraphs: List[Dict[str, str]],
    similarity_threshold: float = 0.75,
    distance_threshold: float = 1.5,
):
    texts = [p["text"] for p in paragraphs]
    embeddings = await asyncio.gather(*[create_embedding(text) for text in texts])
    embeddings = np.vstack(embeddings)

    clusters, _ = clustering_paragraphs(
        embeddings, similarity_threshold, distance_threshold
    )

    cluster_info = {}
    for cluster_id, indices in clusters.items():
        cluster_name = f"cluster_{cluster_id}"
        cluster_info[cluster_name] = {
            "paragraphs": [
                {"text": paragraphs[i]["text"], "url": paragraphs[i]["url"]}
                for i in indices
            ]
        }

    return cluster_info


# Main runner
async def main():
    await initialize_embedding_utils()

    file_path = "no_api/chunking/sem_len/output_dedup.json"  # NDJSON input
    output_path = "no_api/chunking/cluster/sem_len_cluster.json"

    paragraphs = load_paragraphs_with_source(file_path)
    clustered = await cluster_paragraphs_only(paragraphs)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clustered, f, indent=4, ensure_ascii=False)

    print(f"Clustering completed. Output saved to {output_path}")


# Entry point
if __name__ == "__main__":
    asyncio.run(main())
