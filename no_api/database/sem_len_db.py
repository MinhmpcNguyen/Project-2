import json

import numpy as np
import torch
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "intfloat/e5-large-v2"
METADATA_PATH = "no_api/results/sem_len/vector_metadata.json"

INDEX_NAME = "sem-len-index"
PINECONE_API_KEY = (
    "pcsk_645zd_3xo6K25h426g7Pyq9sZfWKymiy5AVGiFBeTUxTWWvHm6UTd15LvyTY51hfyww3V"
)
PINECONE_REGION = "us-east-1"
VECTOR_DIM = 1024

pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
    )

index = pc.Index(INDEX_NAME)

#  Load E5 model
print("Loading embedding model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


def get_embedding(text: str) -> np.ndarray:
    text = "passage: " + text.strip()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.inference_mode():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0]
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy().squeeze().astype(np.float32)


#  Upload từ file metadata
def upload_from_metadata(metadata_path: str):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)

    for meta in tqdm(metadata_list, desc="Uploading metadata vectors to Pinecone"):
        uid = meta["id"]
        text = meta["text"].strip()
        url = meta.get("url", "")
        chunk_index = meta.get("chunk_index", 0)

        if not text:
            continue

        vec = get_embedding(text)
        index.upsert(
            [
                (
                    uid,
                    vec.tolist(),
                    {"text": text, "url": url, "chunk_index": chunk_index},
                )
            ]
        )


#  Run
if __name__ == "__main__":
    upload_from_metadata(METADATA_PATH)
    print(" All metadata vectors uploaded to Pinecone.")
