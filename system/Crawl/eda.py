import json


def analyze_text_lengths(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lengths = [len(item["text"].split()) for item in data]

    if not lengths:
        print("⚠️ No text entries found.")
        return

    print(f"Total entries: {len(lengths)}")
    print(f"Min length: {min(lengths)} words")
    print(f"Max length: {max(lengths)} words")
    print(f"Average length: {sum(lengths) / len(lengths):.2f} words")

    # Optional: Hiển thị histogram
    try:
        import matplotlib.pyplot as plt

        plt.hist(lengths, bins=20, edgecolor="black")
        plt.title("Distribution of Text Lengths (in Words)")
        plt.xlabel("Word Count")
        plt.ylabel("Number of Chunks")
        plt.grid(True)
        plt.show()
    except ImportError:
        print("📉 Install matplotlib to see histogram.")


# Gọi hàm với đường dẫn tới file metadata
analyze_text_lengths("system/Crawl/no_api/results/trava_finance/vector_metadata.json")
