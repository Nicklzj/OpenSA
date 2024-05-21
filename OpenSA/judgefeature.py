import os

file_path = "my_featurelist.txt"

if os.path.exists(file_path):
    # 如果文件存在，则读取每一行的内容到列表中
    with open(file_path, "r") as file:
        feature_list = file.readlines()
        # 去除每一行末尾的换行符
        feature_list = [int(line.strip()) for line in feature_list]
    print("文件存在，内容已读取到列表中。")
    print("列表内容:", feature_list)
else:
    print("文件不存在。")
