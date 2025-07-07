import json

import numpy as np
import torch
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "intfloat/e5-large-v2"
METADATA_PATH = "no_api/results/sem_len_cluster_sem/vector_metadata.json"
INDEX_NAME = "sem-len-cluster-index"
PINECONE_API_KEY = (
    "pcsk_645zd_3xo6K25h426g7Pyq9sZfWKymiy5AVGiFBeTUxTWWvHm6UTd15LvyTY51hfyww3V"
)
PINECONE_REGION = "us-east-1"
VECTOR_DIM = 1024

#  Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
    )

index = pc.Index(INDEX_NAME)

#  Load model
print("Loading E5 embedding model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


def get_embedding(text: str) -> np.ndarray:
    text = "passage: " + text.strip()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.inference_mode():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0]
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    return embedding[0].cpu().numpy().astype(np.float32)


#  Load metadata
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata_list = json.load(f)

#  Upload to Pinecone
for item in tqdm(metadata_list, desc="Uploading metadata to Pinecone"):
    uid = item["id"]
    text = item["text"].strip()
    if not text:
        continue

    vector = get_embedding(text)

    index.upsert(
        [
            (
                uid,
                vector.tolist(),
                {
                    "text": text,
                    "url": item.get("url", []),
                    "chunk_index": item.get("chunk_index", -1),
                    "cluster": item.get("cluster", ""),
                },
            )
        ]
    )

print(" Hoàn tất upload các vector lên Pinecone từ metadata.")
