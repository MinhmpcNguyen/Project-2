import json

import matplotlib.pyplot as plt

# Đọc file JSON từ đường dẫn
with open(
    "system/Crawl/crawl_results_json/trava_finance.json", "r", encoding="utf-8"
) as f:
    data = json.load(f)

# Phân tích độ dài từng đoạn content
lengths = []
urls = []

for page in data:
    url = page["url"]
    for content in page["content"]:
        length = len(content.split())  # hoặc len(content) nếu muốn đếm theo số ký tự
        lengths.append(length)
        urls.append(url)

#  In ra thống kê cơ bản
print(f"Tổng số đoạn: {len(lengths)}")
print(f"Độ dài trung bình: {sum(lengths) / len(lengths):.2f} từ")
print(f"Độ dài lớn nhất: {max(lengths)} từ")
print(f"Độ dài nhỏ nhất: {min(lengths)} từ")

#  Vẽ biểu đồ phân phối
plt.figure(figsize=(10, 6))
plt.hist(lengths, bins=30, color="skyblue", edgecolor="black")
plt.title("Phân phối độ dài của các đoạn content")
plt.xlabel("Số từ mỗi đoạn")
plt.ylabel("Số lượng đoạn")
plt.grid(True)
plt.show()
