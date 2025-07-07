import json
import time

import numpy as np
import pandas as pd
import torch
from pinecone import Pinecone
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

CSV_PATH = "no_api/processed_ground_truth_clean.csv"
PINECONE_API_KEY = (
    "pcsk_645zd_3xo6K25h426g7Pyq9sZfWKymiy5AVGiFBeTUxTWWvHm6UTd15LvyTY51hfyww3V"
)
PINECONE_ENV = "us-east-1"
INDEX_NAME = "len1-index"
TOP_K = 50
SIM_THRESHOLD = 0.85
MODEL_NAME = "intfloat/e5-large-v2"
OUTPUT_CSV = "len_dense_dynamic_e5_pinecone1.csv"


#  Load embedding model
print(" Loading E5 model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


def embed_e5(text: str) -> np.ndarray:
    text = "query: " + text.strip()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0]
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy().astype(np.float32)


#  Pinecone setup
print(" Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


#  Determine threshold dynamically
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
    elif skewness < -0.5:
        return min(n, max(base_k - 1, 1))


#  Search
def dense_search_with_dynamic_threshold(query: str, top_k=TOP_K):
    vec = embed_e5(query).tolist()
    res = index.query(vector=vec, top_k=top_k, include_metadata=True)

    raw_results = []
    for match in res["matches"]:
        meta = match.get("metadata", {})
        raw_results.append(
            {
                "id": match.get("id", ""),
                "text": meta.get("text", ""),
                "score": match.get("score", 0.0),
                "chunk_index": int(meta.get("chunk_index", -1)),
                "url": meta.get("url", ""),
            }
        )

    similarities = [r["score"] for r in raw_results]
    threshold = determine_threshold_strategy(similarities)

    return raw_results[:threshold]


#  Load questions
df = pd.read_csv(CSV_PATH)
df["Contents"] = df["Contents"].apply(lambda x: json.loads(x.replace("'", '"')))

#  Evaluate
results = []
start = time.time()

for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
    question = row["Question"]
    gt_contents = row["Contents"]

    retrieved = dense_search_with_dynamic_threshold(question)
    retrieved_texts = [r["text"] for r in retrieved]
    retrieved_urls = [r["url"] for r in retrieved]

    sims = []
    for gt in gt_contents:
        vec_gt = embed_e5(gt)
        for r_text in retrieved_texts:
            vec_r = embed_e5(r_text)
            sims.append(cosine_similarity(vec_gt, vec_r)[0][0])

    max_sim = np.max(sims) if sims else 0
    is_correct = max_sim >= SIM_THRESHOLD

    results.append(
        {
            "Question": question,
            "Max Cosine Similarity": round(max_sim, 4),
            "Correct (>=0.85)": is_correct,
            "GroundTruths": gt_contents,
            "RetrievedResults": retrieved_texts,
            "TopURLs": retrieved_urls,
            "Source": row["URLs"],
        }
    )

#  Save result with progress
df_result = pd.DataFrame(results)
import csv

with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=df_result.columns)
    writer.writeheader()
    for row in tqdm(results, desc="Saving CSV"):
        writer.writerow(row)
