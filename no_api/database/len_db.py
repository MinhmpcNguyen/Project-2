import json

import numpy as np
import torch
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ===  CONFIG ===
MODEL_NAME = "intfloat/e5-large-v2"
METADATA_JSON = "no_api/results/len/vector_metadata1.json"  # sử dụng file metadata đã chứa id cố định
INDEX_NAME = "len1-index"
PINECONE_API_KEY = (
    "pcsk_645zd_3xo6K25h426g7Pyq9sZfWKymiy5AVGiFBeTUxTWWvHm6UTd15LvyTY51hfyww3V"
)
PINECONE_REGION = "us-east-1"
# ==================

#  Khởi tạo Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
    )

index = pc.Index(INDEX_NAME)

#  Load model
print("Loading embedding model...")
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


#  Load metadata (bao gồm id đã tạo sẵn)
with open(METADATA_JSON, "r", encoding="utf-8") as f:
    metadata_list = json.load(f)

#  Upload từng entry lên Pinecone
for item in tqdm(metadata_list, desc="Uploading to Pinecone"):
    text = item["text"]
    vec = get_embedding(text)

    index.upsert(
        [
            (
                item["id"],
                vec.tolist(),
                {
                    "text": text,
                    "url": item.get("url", ""),
                    "chunk_index": item.get("chunk_index", -1),
                },
            )
        ]
    )

print(" Hoàn tất upload lên Pinecone.")
