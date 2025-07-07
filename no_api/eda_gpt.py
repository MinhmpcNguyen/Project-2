import matplotlib.pyplot as plt
import pandas as pd

# Đọc dữ liệu
df = pd.read_csv("no_api/results_with_dense_similarity.csv")

# Thiết lập kích thước và spacing
fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Trục x là số câu hỏi từ 1 đến n
x = range(1, len(df) + 1)

# Biểu đồ 1: similarity khi có liên kết
axs[0].bar(x, df["max_sim_with_link_dense"], color="skyblue")
axs[0].set_title("ChatGPT Max Similarity Score With Link (Dense)")
axs[0].set_ylabel("Similarity Score")
axs[0].set_ylim(0, 1)
axs[0].axhline(
    y=0.85, color="red", linestyle="--", linewidth=1.5, label="Threshold = 0.85"
)
axs[0].legend()

# Biểu đồ 2: similarity khi không có liên kết
axs[1].bar(x, df["max_sim_without_link_dense"], color="salmon")
axs[1].set_title("ChatGPT Max Similarity Score Without Link (Dense)")
axs[1].set_xlabel("Question Number")
axs[1].set_ylabel("Similarity Score")
axs[1].set_ylim(0, 1)
axs[1].axhline(
    y=0.85, color="red", linestyle="--", linewidth=1.5, label="Threshold = 0.85"
)
axs[1].legend()

# Hiển thị các nhãn trục x thưa ra để không bị chồng lên nhau
plt.xticks(ticks=x, labels=x, rotation=90)
plt.tight_layout()
plt.show()
