import asyncio
import json
from typing import List

import numpy as np
import tiktoken
import torch
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity as cosine_sim
from transformers import AutoModel, AutoTokenizer

# Load environment variables
load_dotenv()

# Globals
embedding_cache = {}
tokenizer = None
model = None
encoding = tiktoken.get_encoding("cl100k_base")


async def initialize_embedding_utils():
    global tokenizer, model
    model_name = "intfloat/e5-large-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return {"embedding": model_name}


async def create_embedding(paragraph: str):
    if paragraph in embedding_cache:
        return embedding_cache[paragraph]

    text = "query: " + paragraph.strip()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0]
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    result = embedding.cpu().numpy().squeeze()
    embedding_cache[paragraph] = result
    return result


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    vec_a = vec_a.reshape(1, -1)
    vec_b = vec_b.reshape(1, -1)
    return float(cosine_sim(vec_a, vec_b)[0][0])


async def compute_advanced_similarities(paragraphs: List[str], lookahead: int = 8):
    if len(paragraphs) < 2:
        return {"similarities": [], "average": 0.0, "variance": 0.0}

    embeddings = await asyncio.gather(*[create_embedding(p) for p in paragraphs])
    similarities = []
    similarity_sum = 0

    for i in range(len(embeddings) - 1):
        max_similarity = cosine_similarity(embeddings[i], embeddings[i + 1])
        for j in range(i + 2, min(i + lookahead + 1, len(embeddings))):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            max_similarity = max(max_similarity, sim)
        similarities.append(max_similarity)
        similarity_sum += max_similarity

    avg = similarity_sum / len(similarities)
    var = np.var(similarities)
    return {"similarities": similarities, "average": avg, "variance": var}


def adjust_threshold(
    average: float,
    variance: float,
    base_threshold: float = 0.4,
    lower_bound: float = 0.2,
    upper_bound: float = 0.8,
    variance_lower: float = 0.01,
    variance_upper: float = 0.05,
    average_lower: float = 0.3,
    average_upper: float = 0.7,
    decrease_by: float = 0.1,
    increase_by: float = 0.1,
):
    threshold = base_threshold
    if variance < variance_lower:
        threshold -= decrease_by
    elif variance > variance_upper:
        threshold += increase_by

    if average < average_lower:
        threshold += increase_by / 2
    elif average > average_upper:
        threshold -= decrease_by / 2

    return min(max(threshold, lower_bound), upper_bound)


async def rechunk_clustered_data(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        cluster_data = json.load(f)

    print("Rechunking clusters...")
    result = {}

    for cluster_name, cluster_info in cluster_data.items():
        raw_paragraphs = cluster_info.get("paragraphs", [])
        if not raw_paragraphs:
            continue

        paragraphs = [p["text"] for p in raw_paragraphs]
        urls = [p["url"] for p in raw_paragraphs]

        sim_result = await compute_advanced_similarities(paragraphs)
        threshold = adjust_threshold(sim_result["average"], sim_result["variance"])

        chunks = []
        current_chunk = [paragraphs[0]]
        current_urls = [urls[0]]
        chunk_length = len(encoding.encode(paragraphs[0]))

        for i in range(1, len(paragraphs)):
            next_paragraph = paragraphs[i]
            next_url = urls[i]
            next_length = len(encoding.encode(next_paragraph))
            similarity = (
                sim_result["similarities"][i - 1]
                if sim_result["similarities"]
                else None
            )
            combined_length = chunk_length + next_length

            should_merge = (
                (similarity is None or similarity >= threshold)
                and (combined_length <= 1024 or chunk_length < 50 or next_length < 50)
                and combined_length <= 1024
            )

            if should_merge:
                current_chunk.append(next_paragraph)
                current_urls.append(next_url)
                chunk_length += next_length
            else:
                chunks.append(
                    {
                        "content": " ".join(current_chunk).strip(),
                        "source_urls": list(set(current_urls)),
                    }
                )
                current_chunk = [next_paragraph]
                current_urls = [next_url]
                chunk_length = next_length

        if current_chunk:
            chunks.append(
                {
                    "content": " ".join(current_chunk).strip(),
                    "source_urls": list(set(current_urls)),
                }
            )

        result[cluster_name] = {"Chunks": chunks}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f" Saved rechunked output to: {output_path}")


async def main():
    await initialize_embedding_utils()
    await rechunk_clustered_data(
        input_path="no_api/chunking/cluster/sem_len_cluster.json",
        output_path="no_api/chunking/cluster/sem_len_cluster_rechunked.json",
    )


if __name__ == "__main__":
    asyncio.run(main())
