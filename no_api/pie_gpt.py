import matplotlib.pyplot as plt
import pandas as pd

# Đọc dữ liệu
df = pd.read_csv("no_api/results_with_dense_similarity.csv")

# Đếm số lượng câu được trả lời và không được trả lời
with_link_counts = df["with_link_answered"].value_counts()
without_link_counts = df["without_link_answered"].value_counts()

# Chuẩn bị dữ liệu
labels = ["Answered", "Not Answered"]
with_link_sizes = [with_link_counts.get(True, 0), with_link_counts.get(False, 0)]
without_link_sizes = [
    without_link_counts.get(True, 0),
    without_link_counts.get(False, 0),
]
colors = ["lightblue", "lightgray"]

# Vẽ biểu đồ
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Pie chart for with_link_answered
axs[0].pie(
    with_link_sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90
)
axs[0].set_title("GPT With Link Answered")

# Pie chart for without_link_answered
axs[1].pie(
    without_link_sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90
)
axs[1].set_title("GPT Without Link Answered")

plt.tight_layout()
plt.show()
