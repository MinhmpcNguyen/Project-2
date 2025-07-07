import matplotlib.pyplot as plt
import pandas as pd

# Đọc dữ liệu từ file CSV
df = pd.read_csv("LLM_len_hybrid_dynamic_e5_consistent_id_max1.csv")

df["Question Number"] = range(1, len(df) + 1)

# Vẽ biểu đồ
plt.figure(figsize=(20, 6))
plt.bar(
    df["Question Number"], df["Max Cosine Similarity"], width=0.5, color="steelblue"
)
plt.xlabel("Question Number")
plt.ylabel("Max Cosine Similarity")
plt.title("System Max Cosine Similarity per Question")

# Thêm đường ngang tại ngưỡng 0.85
plt.axhline(
    y=0.85, color="red", linestyle="--", linewidth=1.5, label="Threshold = 0.85"
)

# Hiển thị nhãn trục X xoay 90 độ
plt.xticks(df["Question Number"], rotation=90)

# Lưới trục y
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
