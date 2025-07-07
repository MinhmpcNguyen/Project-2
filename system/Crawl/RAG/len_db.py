import numpy as np
import torch
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "intfloat/e5-large-v2"
PINECONE_API_KEY = (
    "pcsk_645zd_3xo6K25h426g7Pyq9sZfWKymiy5AVGiFBeTUxTWWvHm6UTd15LvyTY51hfyww3V"
)
PINECONE_REGION = "us-east-1"


#  Init Pinecone + model once
pc = Pinecone(api_key=PINECONE_API_KEY)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


def get_embedding(text: str) -> np.ndarray:
    text = "passage: " + text.strip()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0]
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    return embedding[0].cpu().numpy().astype(np.float32)


def upload_to_pinecone(metadata_list: list, index_name: str):
    # Tạo Pinecone index nếu chưa tồn tại
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
        )

    index = pc.Index(index_name)

    for item in metadata_list:
        vec = get_embedding(item["text"])
        index.upsert(
            [
                (
                    item["id"],
                    vec.tolist(),
                    {
                        "text": item["text"],
                        "url": item.get("url", ""),
                        "chunk_index": item.get("chunk_index", -1),
                    },
                )
            ]
        )
