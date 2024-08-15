import matplotlib.pyplot as plt

# 创建数据
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# 绘制折线图
plt.plot(x, y, marker='o', linestyle='-')

# 添加标题和标签
plt.title('Prime Numbers')
plt.xlabel('Index')
plt.ylabel('Value')

# 显示网格线
plt.grid(True)

# 显示图形
plt.show()
