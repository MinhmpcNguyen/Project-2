import asyncio
import json
import time
from typing import List, Optional

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity as cosine_sim

# Load environment variables
load_dotenv()

# Global variables
embedding_cache = {}
azure_embedder = None
azure_chat_model = None


import torch
from transformers import AutoModel, AutoTokenizer

embedding_cache = {}
tokenizer = None
model = None


async def initialize_embedding_utils():
    global tokenizer, model
    model_name = "intfloat/e5-large-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return {"embedding": "intfloat/e5-large-v2"}


async def create_embedding(text: str):
    if text in embedding_cache:
        return embedding_cache[text]

    text = "query: " + text.strip()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0]  # CLS token
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    result = embedding.cpu().numpy().squeeze()
    embedding_cache[text] = result
    return result


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    vec_a = vec_a.reshape(1, -1)
    vec_b = vec_b.reshape(1, -1)
    return float(cosine_sim(vec_a, vec_b)[0][0])


async def compute_advanced_similarities(
    paragraphs: List[str],
    num_similarity_paragraphs_lookahead: int = 8,
    logging: bool = False,
):
    if len(paragraphs) < 2:
        return {"similarities": [], "average": 0.0, "variance": 0.0}

    embeddings = await asyncio.gather(
        *[create_embedding(paragraph) for paragraph in paragraphs]
    )
    similarities = []
    similarity_sum = 0

    for i in range(len(embeddings) - 1):
        max_similarity = cosine_similarity(embeddings[i], embeddings[i + 1])

        if logging:
            print(f"\nSimilarity scores for paragraph {i}:")
            print(f"Base similarity with next paragraph: {max_similarity}")

        for j in range(
            i + 2, min(i + num_similarity_paragraphs_lookahead + 1, len(embeddings))
        ):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if logging:
                print(f"Similarity with paragraph {j}: {sim}")
            max_similarity = max(max_similarity, sim)

        similarities.append(max_similarity)
        similarity_sum += max_similarity

    if len(similarities) == 0:
        return {"similarities": [], "average": 0.0, "variance": 0.0}

    average = similarity_sum / len(similarities)
    variance = np.var(similarities)

    return {"similarities": similarities, "average": average, "variance": variance}


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
    if lower_bound >= upper_bound:
        raise ValueError("Invalid bounds: lower_bound must be less than upper_bound.")

    adjusted_threshold = base_threshold
    if variance < variance_lower:
        adjusted_threshold -= decrease_by
    elif variance > variance_upper:
        adjusted_threshold += increase_by

    if average < average_lower:
        adjusted_threshold += increase_by / 2
    elif average > average_upper:
        adjusted_threshold -= decrease_by / 2

    return min(max(adjusted_threshold, lower_bound), upper_bound)


import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")


async def create_chunks(
    paragraphs: List[str],
    similarities: Optional[List[float]],
    similarity_threshold: float,
    max_length: int = 1024,
    min_length: int = 50,
    ultimate_max_length: int = 2048,
    logging: bool = False,
) -> List[str]:
    chunks = []
    current_chunk = [paragraphs[0]]
    chunk_length = len(encoding.encode(paragraphs[0]))

    for i in range(1, len(paragraphs)):
        next_paragraph = paragraphs[i]
        next_length = len(encoding.encode(next_paragraph))
        similarity = similarities[i - 1] if similarities else None
        combined_length = chunk_length + next_length

        should_merge = (
            (similarity is None or similarity >= similarity_threshold)
            and (
                combined_length <= max_length
                or chunk_length < min_length
                or next_length < min_length
            )
            and combined_length <= ultimate_max_length
        )

        if logging:
            print(f"\n--- Paragraph {i} ---")
            print(f"Similarity: {similarity}")
            print(f"Chunk length: {chunk_length}")
            print(f"Next length: {next_length}")
            print(f"Combined length: {combined_length}")
            print(f"Should merge: {should_merge}")

        if should_merge:
            current_chunk.append(next_paragraph)
            chunk_length += next_length
        else:
            chunks.append(" ".join(current_chunk).strip())
            current_chunk = [next_paragraph]
            chunk_length = next_length

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks


def remove_duplicate_chunks(data):
    """Loại bỏ chunks trùng lặp trên toàn bộ tập dữ liệu, nhưng chỉ nếu chúng xuất hiện trong nhiều tài liệu khác nhau."""
    seen_chunks = {}  # Sử dụng dict để lưu URL đầu tiên của mỗi chunk
    filtered_data = []

    for doc in data:
        new_chunks = []
        for chunk in doc["Chunks"]:
            content = chunk["content"].strip()

            # Kiểm tra nếu chunk này đã tồn tại trong một tài liệu trước đó
            if content in seen_chunks:
                # Nếu chunk đã được lưu trong một tài liệu khác, bỏ qua chunk này
                print(
                    f"Removing duplicate chunk in {doc['Url']} (already exists in {seen_chunks[content]})"
                )
                continue

            # Nếu chunk chưa tồn tại, thêm vào bộ nhớ
            seen_chunks[content] = doc["Url"]
            new_chunks.append(chunk)

        if new_chunks:
            filtered_data.append({"Url": doc["Url"], "Chunks": new_chunks})

    return filtered_data


from tqdm import tqdm


async def main():
    import nltk
    from nltk.tokenize import sent_tokenize

    nltk.download("punkt")

    start_time = time.time()
    print("Starting data processing...")

    await initialize_embedding_utils()

    input_file = "no_api/reformat/processed_results/final_output.json"
    output_file = "no_api/chunking/sem_len/sem_len.json"

    print("Loading input data...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f" Input data loaded in {time.time() - start_time:.2f} seconds.")

    output_stream = open(output_file, "w", encoding="utf-8", newline="\n")

    for doc in tqdm(data, desc="Processing documents"):
        raw_paragraphs = doc.get("content", [])
        if not raw_paragraphs:
            print(f"Skipping {doc['url']} - No content found.")
            continue

        # Tách toàn bộ đoạn văn thành các câu
        sentences = []
        for para in raw_paragraphs:
            if isinstance(para, str):
                sentences.extend(sent_tokenize(para))
            elif isinstance(para, list):
                for sub_para in para:
                    if isinstance(sub_para, str):
                        sentences.extend(sent_tokenize(sub_para))

        if not sentences:
            print(f"Skipping {doc['url']} - No sentences extracted.")
            continue

        similarity_results = await compute_advanced_similarities(sentences)
        threshold = adjust_threshold(
            similarity_results["average"], similarity_results["variance"]
        )

        chunks = await create_chunks(
            sentences, similarity_results["similarities"], threshold
        )

        chunk_list = [{"content": chunk} for chunk in chunks]
        result = {"Url": doc["url"], "Chunks": chunk_list}

        json.dump(result, output_stream, ensure_ascii=False)
        output_stream.write("\n")
        output_stream.flush()

    output_stream.close()
    print(f"Streaming output saved to {output_file}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds.")


#  ** Chạy Script**
if __name__ == "__main__":
    asyncio.run(main())


# 71.66s
