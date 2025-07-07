import matplotlib.pyplot as plt
import pandas as pd

# Đọc dữ liệu
df = pd.read_csv("LLM_len_hybrid_dynamic_e5_consistent_id_max1.csv")

# Đếm số lượng đúng và sai
counts = df["Correct (>=0.85)"].value_counts()
labels = ["Correct", "Incorrect"]
sizes = [counts.get(True, 0), counts.get(False, 0)]
colors = ["lightgreen", "lightcoral"]

# Vẽ pie chart
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
plt.title("Hệ thống: Tỷ lệ câu trả lời đúng và sai theo ngưỡng 0.85")
plt.axis("equal")  # Đảm bảo hình tròn
plt.show()
