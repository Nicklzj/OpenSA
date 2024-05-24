import os
import csv

# 定义一个函数来处理 CSV 文件
def process_csv(file_path, label):
    result = {"x": [], "y": label}
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
        if rows[0][0].isdigit():
            result["x"] = [row[1] for row in rows]
        else:
            del rows[0]
            result["x"] = [row[1] for row in rows]
    return result

# 获取当前目录下所有文件夹中的 CSV 文件
current_dir = os.getcwd()
result_csv = []

for root, dirs, files in os.walk(current_dir):
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            label = os.path.basename(root)
            result_csv.append(process_csv(file_path, label))

# 将处理后的结果写入到 Result.csv 文件中
output_file = os.path.join(current_dir, "Result.csv")
with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ['x', 'y']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in result_csv:
        writer.writerow(row)
