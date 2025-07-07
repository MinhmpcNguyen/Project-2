import json

import numpy as np
import torch
from google import genai
from pinecone import Pinecone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

METADATA_PATH = "no_api/results/len/vector_metadata.json"
PINECONE_API_KEY = (
    "pcsk_645zd_3xo6K25h426g7Pyq9sZfWKymiy5AVGiFBeTUxTWWvHm6UTd15LvyTY51hfyww3V"
)
PINECONE_ENV = "us-east-1"
INDEX_NAME = "len-index"
TOP_K = 50
FINAL_K = 5
SIM_THRESHOLD = 0.85
MODEL_NAME = "intfloat/e5-large-v2"
GEMINI_API_KEY = "AIzaSyD9ALqpcywYTa1gbNRdKuy0iTzu5OpDNz8"


# Load embedding model
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


# Pinecone setup
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load metadata
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    all_metadata = json.load(f)

id2meta = {item["id"]: item for item in all_metadata}
all_texts = [item["text"] for item in all_metadata]

vectorizer = TfidfVectorizer().fit(all_texts)
corpus_sparse = vectorizer.transform(all_texts)


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
    vec = vectorizer.transform([query])
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


def hybrid_search(query: str):
    dense = dense_search(query, TOP_K)
    sparse = sparse_search(query, TOP_K)
    fused = rrf_fusion([(d["id"], d["score"]) for d in dense], sparse, FINAL_K)

    results = []
    for match_id, _ in fused:
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


def generate_content_with_gemini(
    prompt: str, model: str = "gemini-2.0-flash", api_key: str = None
) -> str:
    if api_key is None:
        raise ValueError("API key must be provided.")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return response.text


#  Ask 1 question
question = "What is Trava Finance?"
ground_truths = [
    "TRAVA FINANCE is the world’s first decentralized marketplace for cross-chain lending. Different from existing approaches, we offer a flexible mechanism in which users can create and manage their own lending pools to start a lending business. /// Trava Finance is a DeFi protocol that aims to become an AI-driven Lending Station, that provides unique Lending services."
]

retrieved = hybrid_search(question)
retrieved_texts = [(r["text"], r["url"]) for r in retrieved]

# Build context and source references
context_texts = ""
url_refs = []
for i, (text, url_list) in enumerate(retrieved_texts):
    index = i + 1
    source_url = url_list if url_list else "N/A"
    context_texts += f"{index}. {text}\n"
    url_refs.append(f"[{index}]: {source_url}")

PROMPT = f"""You are a helpful assistant. Use the context below to answer the user's question.
When possible, cite your sources inline using [number] and list them at the end.

Context:
{context_texts}

Question: {question}
Answer:

Please also include the reference list at the end like this:
{chr(10).join(url_refs)}
"""

llm_answer = generate_content_with_gemini(PROMPT, api_key=GEMINI_API_KEY)

# Print result
print("\n Gemini Answer:\n", llm_answer)

print("\n Sources:")
for ref in url_refs:
    print(ref)

# Evaluate similarity
retrieved_vectors = [r["vector"] for r in retrieved]
vec_llm = embed_dense(llm_answer)
sims = []
for gt in ground_truths:
    vec_gt = embed_dense(gt)
    sim = cosine_similarity(vec_gt, vec_llm)[0][0]
    sims.append(sim)

max_sim = np.max(sims) if sims else 0

print(f"\n Max Cosine Similarity: {max_sim:.4f}")
print(" Is Correct (>= 0.85):", max_sim >= SIM_THRESHOLD)
