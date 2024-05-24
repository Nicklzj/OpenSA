import csv

# 打开原始CSV文件
with open('result.csv', 'r', newline='') as infile:
    reader = csv.reader(infile)
    
    # 创建新的CSV文件
    with open('test_output.csv', 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        
        # 遍历原始文件的每一行
        for row in reader:
            # 分割第一个单元格中的列表
            numbers = row[0].strip('[]').split(',')
            
            # 将列表中的每个元素写入新文件
            writer.writerow(numbers + [row[1]])
