import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.decomposition import PCA  
from sklearn.datasets import load_iris  # 假设我们使用鸢尾花数据集作为示例  
  
# 加载数据集  
X, _ = load_iris(return_X_y=True)  
  
# 计算不同主成分数量下的累积解释方差比例  
explained_variances = []  
for n_comp in range(1, X.shape[1] + 1):  # 从1到特征数量  
    pca = PCA(n_components=n_comp)  
    pca.fit(X)  
    explained_variances.append(np.sum(pca.explained_variance_ratio_))  
  
# 绘制累积解释方差比例图  
plt.figure(figsize=(10, 6))  
plt.plot(range(1, len(explained_variances) + 1), explained_variances, marker='o')  
plt.xlabel('Number of components')  
plt.ylabel('Cumulative explained variance')  
plt.title('Cumulative Explained Variance as a Function of the Number of Principal Components')  
plt.grid(True)  
plt.show()