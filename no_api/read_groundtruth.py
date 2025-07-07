import json
import re

import pandas as pd

# Đọc file CSV gốc
file_path = "trava rag test 27_02 - Trang tính1.csv"
df = pd.read_csv(file_path)

# Danh sách mới để lưu dòng hợp lệ
valid_rows = []

# Duyệt từng dòng
for _, row in df.iterrows():
    cell = row.get("Ground truth (url and content)", "")
    try:
        # Tìm các object JSON nhỏ trong chuỗi
        json_objs = re.findall(r"\{.*?\}", str(cell).replace("\n", " "))
        urls = []
        contents = []
        for obj_str in json_objs:
            obj = json.loads(obj_str.replace("'", '"'))
            urls.append(obj.get("url", ""))
            contents.append(obj.get("content", ""))
        # Nếu ít nhất một URL và content tồn tại → giữ dòng
        if urls and contents:
            row["URLs"] = urls
            row["Contents"] = contents
            valid_rows.append(row)
    except Exception:
        continue  # Bỏ qua dòng bị lỗi

# Tạo DataFrame mới chỉ gồm dòng hợp lệ
df_clean = pd.DataFrame(valid_rows)

# Lưu ra file mới nếu muốn
df_clean.to_csv("processed_ground_truth_clean.csv", index=False)

# Hiển thị vài dòng kết quả
print(df_clean[["Ground truth (url and content)", "URLs", "Contents"]].head())
