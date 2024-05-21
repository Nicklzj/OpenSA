from mpl_toolkits.mplot3d import Axes3D 
from sklearn.datasets import make_swiss_roll 
from sklearn.decomposition import KernelPCA 
import matplotlib.pyplot as plt
 # 生成三维非线性数据：瑞士卷数据集 
X, color = make_swiss_roll(n_samples=800, noise=0.05) 
# 使用核PCA进行数据降维（映射到二维）
kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=0.1)
X_kpca = kpca.fit_transform(X) 

# 可视化结果 
fig = plt.figure(figsize=(12, 6))

 # 原始三维数据 
ax = fig.add_subplot(1, 2, 1, projection='3d') 
ax.set_title("Original Space (3D)") 
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)

 # 核PCA变换后的二维数据 
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title("Kernel PCA Space (2D)") 
ax2.scatter(X_kpca[:, 0], X_kpca[:, 1], c=color, cmap=plt.cm.Spectral) 
plt.show()