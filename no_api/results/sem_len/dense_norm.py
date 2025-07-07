import csv
import json

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
INDEX_NAME = "sem-len-index"
SIM_THRESHOLD = 0.85
TOP_K = 5
MODEL_NAME = "intfloat/e5-large-v2"
OUTPUT_CSV = "sem_len_dense_norm_e5_pinecone.csv"


#  Load embedding model
print("Loading E5 model...")
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
    return emb.cpu().numpy().squeeze().astype(np.float32)


#  Pinecone init
print("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


#  Search từ Pinecone
def dense_search(query: str, top_k: int = TOP_K):
    query_vec = embed_e5(query).tolist()
    res = index.query(vector=query_vec, top_k=top_k, include_metadata=True)

    results = []
    for match in res["matches"]:
        meta = match.get("metadata", {})
        results.append(
            {
                "text": meta.get("text", ""),
                "vector": embed_e5(meta.get("text", "")),
                "url": meta.get("url", []),
            }
        )
    return results


#  Load câu hỏi
df = pd.read_csv(CSV_PATH)
df["Contents"] = df["Contents"].apply(lambda x: json.loads(x.replace("'", '"')))

#  Đánh giá
results = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="🔍 Evaluating"):
    question = row["Question"]
    content_texts = row["Contents"]

    # Search Pinecone
    retrieved = dense_search(question, top_k=TOP_K)
    retrieved_texts = [r["text"] for r in retrieved]
    retrieved_urls = [r["url"] for r in retrieved]

    # Cosine similarity giữa ground truth & retrieved
    sims = []
    for gt in content_texts:
        vec_gt = embed_e5(gt).reshape(1, -1)
        for r in retrieved:
            vec_r = r["vector"].reshape(1, -1)
            sim = cosine_similarity(vec_gt, vec_r)[0][0]
            sims.append(sim)

    max_sim = np.max(sims) if sims else 0
    is_correct = max_sim >= SIM_THRESHOLD

    results.append(
        {
            "Question": question,
            "Max Cosine Similarity": round(max_sim, 4),
            "Correct (>=0.85)": is_correct,
            "GroundTruths": content_texts,
            "RetrievedResults": retrieved_texts,
            "TopURLs": retrieved_urls,
            "Source": row["URLs"],
        }
    )

#  Save kết quả
df_result = pd.DataFrame(results)
with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=df_result.columns)
    writer.writeheader()
    for row in tqdm(df_result.to_dict(orient="records"), desc="Saving CSV"):
        writer.writerow(row)

print(f" Đã lưu kết quả vào {OUTPUT_CSV}")
