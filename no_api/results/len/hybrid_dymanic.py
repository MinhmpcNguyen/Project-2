import json
import time

import numpy as np
import pandas as pd
import torch
from pinecone import Pinecone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

CSV_PATH = "no_api/processed_ground_truth_clean.csv"
METADATA_PATH = "no_api/results/len/vector_metadata1.json"
PINECONE_API_KEY = (
    "pcsk_645zd_3xo6K25h426g7Pyq9sZfWKymiy5AVGiFBeTUxTWWvHm6UTd15LvyTY51hfyww3V"
)
PINECONE_ENV = "us-east-1"
INDEX_NAME = "len1-index"
TOP_K = 50
FINAL_K = 5
SIM_THRESHOLD = 0.85
MODEL_NAME = "intfloat/e5-large-v2"


#  Load embedding model
print("Loading E5 model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


def embed_dense(text: str) -> np.ndarray:
    text = "query: " + text.strip()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model(**inputs)
        emb = output.last_hidden_state[:, 0]
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy().astype(np.float32)


#  Connect to Pinecone
print("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

#  Load metadata (with ID)
print("Loading metadata...")
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    all_metadata = json.load(f)

id2meta = {item["id"]: item for item in all_metadata}
all_texts = [item["text"] for item in all_metadata]
print(f" Loaded {len(all_metadata)} metadata entries")

#  TF-IDF vectorizer
print("Fitting TF-IDF...")
vectorizer = TfidfVectorizer().fit(all_texts)
corpus_sparse = vectorizer.transform(all_texts)


def embed_sparse(text: str):
    return vectorizer.transform([text])


def dense_search(query: str, top_k: int = TOP_K):
    query_vec = embed_dense(query).tolist()
    res = index.query(vector=query_vec, top_k=top_k, include_metadata=True)

    results = []
    for match in res["matches"]:
        match_id = match.get("id")
        meta = id2meta.get(match_id, {})
        results.append(
            {
                "id": match_id,
                "text": meta.get("text", ""),
                "url": meta.get("url", []),
                "vector": embed_dense(meta.get("text", "")),
                "score": match.get("score", 0.0),
            }
        )
    return results


def sparse_search(query: str, top_k: int = TOP_K):
    vec = embed_sparse(query)
    sims = cosine_similarity(vec, corpus_sparse)[0]
    top_idxs = np.argsort(sims)[::-1][:top_k]
    return [(all_metadata[i]["id"], float(sims[i])) for i in top_idxs]


def rrf_fusion(dense, sparse, top_k):
    scores = {}
    for rank, (id_, _) in enumerate(dense):
        scores[id_] = scores.get(id_, 0) + 1 / (rank + 1)
    for rank, (id_, _) in enumerate(sparse):
        scores[id_] = scores.get(id_, 0) + 1 / (rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


def determine_threshold_strategy(similarities, base_k=5):
    similarities = np.array(similarities)
    mean_sim = np.mean(similarities)
    std_dev_sim = np.std(similarities)
    skewness = (3 * (mean_sim - np.median(similarities))) / (std_dev_sim + 1e-9)
    n = len(similarities)

    if abs(skewness) < 0.5:
        return min(n, base_k)
    elif skewness > 0.5:
        return min(n, base_k + 2)
    else:
        return min(n, max(base_k - 1, 1))


def hybrid_search_dynamic(query: str):
    # 1. Tính điểm từ dense & sparse
    dense = dense_search(query, TOP_K)
    sparse = sparse_search(query, TOP_K)

    # 2. RRF Fusion
    fused = rrf_fusion([(d["id"], d["score"]) for d in dense], sparse, TOP_K)
    scores = [score for _, score in fused]

    # 3. Xác định số lượng kết quả phù hợp
    k_selected = determine_threshold_strategy(scores, base_k=FINAL_K)

    # 4. Lấy top-k theo thứ tự từ fused
    results = []
    for match_id, _ in fused[:k_selected]:
        if match_id in id2meta:
            meta = id2meta[match_id]
            results.append(
                {
                    "text": meta["text"],
                    "url": meta.get("url", []),
                    "vector": embed_dense(meta["text"]),
                }
            )

    return results


#  Evaluate
df = pd.read_csv(CSV_PATH)
df["Contents"] = df["Contents"].apply(lambda x: json.loads(x.replace("'", '"')))

results = []
start = time.time()

for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
    question = row["Question"]
    gt_texts = row["Contents"]

    retrieved = hybrid_search_dynamic(question)
    retrieved_texts = [r["text"] for r in retrieved]
    retrieved_vectors = [r["vector"] for r in retrieved]
    retrieved_urls = [r["url"] for r in retrieved]

    sims = []
    for gt in gt_texts:
        vec_gt = embed_dense(gt)
        for vec_rt in retrieved_vectors:
            sims.append(cosine_similarity(vec_gt, vec_rt)[0][0])

    max_sim = np.max(sims) if sims else 0
    is_correct = max_sim >= SIM_THRESHOLD

    results.append(
        {
            "Question": question,
            "Max Cosine Similarity": round(max_sim, 4),
            "Correct (>=0.85)": is_correct,
            "GroundTruths": gt_texts,
            "RetrievedResults": retrieved_texts,
            "TopURLs": retrieved_urls,
            "Source": row["URLs"],
        }
    )

df_result = pd.DataFrame(results)
df_result.to_csv("len_hybrid_dynamic_e5_consistent_id_max1.csv", index=False)
print(f" Done — {time.time() - start:.2f}s")
