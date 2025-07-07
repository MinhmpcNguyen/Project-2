import ast

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Load data
df = pd.read_csv(
    "Compare to GPT/GT_with_GPT_answer_no_link - processed_ground_truth_clean-2 copy.csv"
)
df["Contents"] = df["Contents"].apply(ast.literal_eval)

# Load embedding model
model_name = "intfloat/e5-large-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


# Hàm encode văn bản
def embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    texts = ["query: " + t for t in texts]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()


# Hàm tính max cosine similarity giữa 1 câu và các đoạn văn
def max_dense_cosine(query, paragraphs):
    if (
        not isinstance(query, str)
        or not isinstance(paragraphs, list)
        or len(paragraphs) == 0
    ):
        return 0.0
    emb_query = embed(query)[0].reshape(1, -1)
    emb_paras = embed(paragraphs)
    sims = cosine_similarity(emb_query, emb_paras).flatten()
    return float(np.max(sims)) if len(sims) > 0 else 0.0


# Tính similarity
tqdm.pandas()
df["max_sim_with_link_dense"] = df.progress_apply(
    lambda row: max_dense_cosine(row["GPT answer with link"], row["Contents"]), axis=1
)
df["max_sim_without_link_dense"] = df.progress_apply(
    lambda row: max_dense_cosine(row["GPT answer without link"], row["Contents"]),
    axis=1,
)
threshold = 0.85
df["with_link_answered"] = df["max_sim_with_link_dense"] > threshold
df["without_link_answered"] = df["max_sim_without_link_dense"] > threshold

df.to_csv("Compare to GPT/results_with_dense_similarity1.csv", index=False)
