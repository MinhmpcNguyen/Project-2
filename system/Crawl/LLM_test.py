import json
import os

import gradio as gr
import numpy as np
import torch
from google import genai
from pinecone import Pinecone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

PINECONE_API_KEY = (
    "pcsk_645zd_3xo6K25h426g7Pyq9sZfWKymiy5AVGiFBeTUxTWWvHm6UTd15LvyTY51hfyww3V"
)
PINECONE_ENV = "us-east-1"
TOP_K = 50
FINAL_K = 5
MODEL_NAME = "intfloat/e5-large-v2"
GEMINI_API_KEY = "AIzaSyD9ALqpcywYTa1gbNRdKuy0iTzu5OpDNz8"

#  Khởi tạo Pinecone và model
pc = Pinecone(api_key=PINECONE_API_KEY)
available_indexes = pc.list_indexes().names()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

#  TF-IDF & Metadata mapping (sẽ load sau khi chọn index)
vectorizer = None
corpus_sparse = None
id2meta = {}
all_metadata = []


def load_index_resources(index_name: str):
    global vectorizer, corpus_sparse, id2meta, all_metadata

    index = pc.Index(index_name)

    #  Đúng logic: chuyển từ '-' → '_' để ra tên thư mục
    metadata_dirname = index_name.replace("-index", "").replace("-", "_")
    metadata_path = (
        f"system/Crawl/no_api/results/{metadata_dirname}/vector_metadata.json"
    )

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found at: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        all_metadata = json.load(f)

    id2meta = {item["id"]: item for item in all_metadata}
    all_texts = [item["text"] for item in all_metadata]
    vectorizer = TfidfVectorizer().fit(all_texts)
    corpus_sparse = vectorizer.transform(all_texts)

    return index


def embed_dense(text: str) -> np.ndarray:
    text = "query: " + text.strip()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model(**inputs)
        emb = output.last_hidden_state[:, 0]
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy().astype(np.float32)


def dense_search(query: str, index, top_k: int = TOP_K):
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


def hybrid_search(query: str, index):
    dense = dense_search(query, index)
    sparse = sparse_search(query)
    fused = rrf_fusion([(d["id"], d["score"]) for d in dense], sparse, FINAL_K)

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


def generate_content_with_gemini(
    prompt: str, model: str = "gemini-2.0-flash", api_key: str = None
) -> str:
    if api_key is None:
        raise ValueError("API key must be provided.")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text


def ask_rag(query: str, index_name: str):
    try:
        index = load_index_resources(index_name)
        retrieved = hybrid_search(query, index)

        context_texts = ""
        url_refs = []
        for i, r in enumerate(retrieved):
            idx = i + 1
            url = r["url"] if r["url"] else "N/A"
            context_texts += f"{idx}. {r['text']} [source: {url}]\n"
            url_refs.append(f"[{idx}]: {url}")

        prompt = f"""You are a helpful assistant. Use the context below to answer the user's question.
When possible, cite your sources inline using [number] and list them at the end.

Context:
{context_texts}

Question: {query}
Answer:

"""
        answer = generate_content_with_gemini(prompt, api_key=GEMINI_API_KEY)
        final_answer = answer.strip() + "\n\nReferences:\n" + "\n".join(url_refs)
        return final_answer, context_texts.strip()

    except FileNotFoundError as e:
        return f"Metadata missing: {str(e)}", "No context retrieved."
    except Exception as e:
        return f"Error: {str(e)}", "No context retrieved."


#  Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Ask your documents — powered by RAG + Gemini")

    index_dropdown = gr.Dropdown(choices=available_indexes, label="Select Index")
    inp = gr.Textbox(label="Your Question", placeholder="E.g. What is Trava Finance?")
    btn = gr.Button("🔍 Ask")
    out_answer = gr.Textbox(label="Gemini Answer", lines=6)
    out_context = gr.Textbox(label="Retrieved Context", lines=10)

    btn.click(
        fn=ask_rag, inputs=[inp, index_dropdown], outputs=[out_answer, out_context]
    )

demo.launch()
