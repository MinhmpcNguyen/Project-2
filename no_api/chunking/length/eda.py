import json

import matplotlib.pyplot as plt


def eda_content_lengths(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    word_lengths = []
    char_lengths = []

    for doc in data:
        for chunk in doc.get("chunks", []):
            content = chunk.get("content", "").strip()
            if content:
                word_lengths.append(len(content.split()))
                char_lengths.append(len(content))

    if not word_lengths:
        print(" No content found.")
        return

    print(f" Total content chunks: {len(word_lengths)}")
    print(
        f"Word count → Min: {min(word_lengths)}, Max: {max(word_lengths)}, Mean: {sum(word_lengths) / len(word_lengths):.2f}"
    )
    print(
        f"Char count → Min: {min(char_lengths)}, Max: {max(char_lengths)}, Mean: {sum(char_lengths) / len(char_lengths):.2f}"
    )

    # Biểu đồ số từ
    plt.figure(figsize=(10, 4))
    plt.hist(word_lengths, bins=10, color="skyblue", edgecolor="black")
    plt.title("Distribution of Content Lengths (Words)")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    # Biểu đồ số ký tự
    plt.figure(figsize=(10, 4))
    plt.hist(char_lengths, bins=10, color="orchid", edgecolor="black")
    plt.title("Distribution of Content Lengths (Characters)")
    plt.xlabel("Character Count")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


# Gọi hàm
eda_content_lengths("no_api/chunking/length/len1.json")
