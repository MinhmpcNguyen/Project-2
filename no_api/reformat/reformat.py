import json
import os
import re

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download("punkt")


# def clean_text(content):
#     """Làm sạch nội dung bằng cách xóa khoảng trắng, xuống dòng dư thừa"""
#     content = re.sub(r"#+", "", content).strip()
#     content = re.sub(r"\n+", " ", content).strip()
#     content = re.sub(r" {2,}", " ", content).strip()
#     return content
def clean_text(content):
    if not isinstance(content, str):
        return ""

    content = re.sub(r"#+", "", content).strip()
    content = re.sub(r"\n+", " ", content).strip()
    content = re.sub(r" {2,}", " ", content).strip()
    return content


def process_json(markdown_pages):
    """
    Đầu vào là list các dict có key: 'url', 'markdown'
    Đầu ra là list các dict có key: 'url', 'content'
    """
    processed = []

    for page in markdown_pages:
        url = page.get("url", "unknown")
        markdown_blocks = page.get("markdown", [])
        unique_contents = set()
        combined_contents = []

        if isinstance(markdown_blocks, list):
            for block in markdown_blocks:
                #  TH1: Nếu block là chuỗi đơn thuần (không có header)
                if isinstance(block, str):
                    text_clean = clean_text(block)
                    if text_clean and text_clean not in unique_contents:
                        unique_contents.add(text_clean)
                        combined_contents.append(text_clean)

                #  TH2: Nếu block là list các dict hoặc list khác
                elif isinstance(block, list):
                    for entry in block:
                        if isinstance(entry, dict):
                            for key, values in entry.items():
                                if isinstance(values, list):
                                    for value in values:
                                        key_clean = clean_text(key)
                                        value_clean = clean_text(value)
                                        combined = f"{key_clean}: {value_clean}"
                                        if combined not in unique_contents:
                                            unique_contents.add(combined)
                                            combined_contents.append(combined)

                #  TH3: Nếu block là dict (có header rõ ràng)
                elif isinstance(block, dict):
                    for key, values in block.items():
                        if isinstance(values, list):
                            for value in values:
                                key_clean = clean_text(key)
                                value_clean = clean_text(value)
                                combined = f"{key_clean}: {value_clean}"
                                if combined not in unique_contents:
                                    unique_contents.add(combined)
                                    combined_contents.append(combined)

        processed.append({"url": url, "content": combined_contents})

    return processed


def split_long_entries(data, word_threshold=80):
    """
    Với mỗi entry trong content, nếu quá dài thì tách thành các câu nhỏ.
    """
    new_data = []

    for doc in data:
        url = doc["url"]
        new_contents = []

        for entry in doc["content"]:
            num_words = len(word_tokenize(entry))
            if num_words > word_threshold:
                # Tách thành nhiều câu nhỏ
                split_sentences = sent_tokenize(entry)
                new_contents.extend(split_sentences)
            else:
                new_contents.append(entry)

        new_data.append({"url": url, "content": new_contents})

    return new_data


def remove_global_duplicates(docs):
    """
    Loại bỏ các nội dung trùng lặp trên toàn bộ dataset.
    Nếu cùng một entry xuất hiện ở nhiều trang, chỉ giữ lại ở trang đầu tiên.
    """
    seen = set()
    for doc in docs:
        filtered_content = []
        for entry in doc["content"]:
            if entry not in seen:
                seen.add(entry)
                filtered_content.append(entry)
        doc["content"] = filtered_content
    return docs


# if __name__ == "__main__":
#     input_file = "crawl_results/results.json"
#     cleaned_output = (
#         "/Users/Yuki/Prj2/no_api/reformat/processed_results/final_output.json"
#     )
#     segmented_output = (
#         "/Users/Yuki/Prj2/no_api/reformat/processed_results/final_output_segmented.json"
#     )

#     os.makedirs(os.path.dirname(cleaned_output), exist_ok=True)

#     # Bước 1: Làm sạch + Gộp key-value
#     with open(input_file, "r", encoding="utf-8") as f:
#         raw_data = json.load(f)

#     processed = process_json(raw_data)

#     # Ghi file trung gian
#     with open(cleaned_output, "w", encoding="utf-8") as f:
#         json.dump(processed, f, indent=4, ensure_ascii=False)

#     print(f" Giai đoạn 1: Đã xử lý xong. Output lưu tại: {cleaned_output}")

#     # Bước 2: Ngắt đoạn quá dài thành câu
#     segmented = split_long_entries(processed, word_threshold=80)

#     with open(segmented_output, "w", encoding="utf-8") as f:
#         json.dump(segmented, f, indent=4, ensure_ascii=False)

#     print(
#         f" Giai đoạn 2: Đã tách đoạn dài thành câu. Output lưu tại: {segmented_output}"
#     )
if __name__ == "__main__":
    input_file = "crawl_results/results.json"
    cleaned_output = (
        "/Users/Yuki/Prj2/no_api/reformat/processed_results/final_output.json"
    )
    segmented_output = (
        "/Users/Yuki/Prj2/no_api/reformat/processed_results/final_output_segmented.json"
    )
    deduplicated_output = (
        "/Users/Yuki/Prj2/no_api/reformat/processed_results/final_output_dedup.json"
    )

    os.makedirs(os.path.dirname(cleaned_output), exist_ok=True)

    # Bước 1: Làm sạch + Gộp key-value
    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    processed = process_json(raw_data)

    with open(cleaned_output, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=4, ensure_ascii=False)

    print(f" Giai đoạn 1: Đã xử lý xong. Output lưu tại: {cleaned_output}")

    # Bước 2: Ngắt đoạn dài
    segmented = split_long_entries(processed, word_threshold=80)

    with open(segmented_output, "w", encoding="utf-8") as f:
        json.dump(segmented, f, indent=4, ensure_ascii=False)

    print(
        f" Giai đoạn 2: Đã tách đoạn dài thành câu. Output lưu tại: {segmented_output}"
    )

    #  Bước 3: Loại bỏ trùng lặp toàn bộ dataset
    deduplicated = remove_global_duplicates(segmented)

    with open(deduplicated_output, "w", encoding="utf-8") as f:
        json.dump(deduplicated, f, indent=4, ensure_ascii=False)

    print(f" Giai đoạn 3: Đã loại bỏ trùng lặp toàn cục. Output: {deduplicated_output}")
