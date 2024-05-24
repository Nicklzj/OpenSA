from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# 加载示例数据集
data = load_iris()
X = data.data
y = data.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练随机森林分类器
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 获取预测概率
probabilities = rf.predict_proba(X_test)

# 查看第一个样本的类别概率
first_sample_prob = probabilities[0]
print(f"类别0的概率: {first_sample_prob[0]}")
print(f"类别1的概率: {first_sample_prob[1]}")
print(f"类别2的概率: {first_sample_prob[2]}")

# 获取每棵树对测试样本的预测结果
tree_predictions = np.array([tree.predict(X_test) for tree in rf.estimators_])

# 查看第一个样本的投票情况
first_sample_votes = tree_predictions[:, 0]
unique, counts = np.unique(first_sample_votes, return_counts=True)
vote_counts = dict(zip(unique, counts))
print(f"第一个样本的投票情况: {vote_counts}")

