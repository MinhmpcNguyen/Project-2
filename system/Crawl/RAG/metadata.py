import json
import os
import uuid


def create_metadata(raw_docs: list, output_dir: str):
    """
    Tạo metadata từ danh sách raw_docs và lưu ra file JSON.

    Args:
        raw_docs (list): Danh sách các tài liệu đã được xử lý từ crawl.
        output_dir (str): Thư mục đầu ra (ví dụ: 'no_api/results/trava_docs').

    Returns:
        metadata_list (list): Danh sách metadata đã tạo.
    """
    metadata_list = []

    for doc in raw_docs:
        url = doc.get("url", "")
        contents = doc.get("content", [])

        for i, content in enumerate(contents):
            content = content.strip()
            if content:
                unique_id = str(uuid.uuid4())
                metadata_list.append(
                    {
                        "id": unique_id,
                        "text": content,
                        "url": url,
                        "chunk_index": i,
                    }
                )

    #  Tạo thư mục nếu chưa có
    os.makedirs(output_dir, exist_ok=True)

    #  Lưu metadata ra file JSON
    output_path = os.path.join(output_dir, "vector_metadata.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)

    print(f" Saved metadata to: {output_path}")
    return metadata_list
