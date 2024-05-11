from sklearn.datasets import load_iris  
from sklearn.decomposition import PCA  
from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC  
from sklearn.metrics import accuracy_score  
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D  
  
# 加载鸢尾花数据集  
iris = load_iris()  
X = iris.data  
y = iris.target  
  
# 划分训练集和测试集  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
  
# 使用PCA进行降维到三维  
pca = PCA(n_components=3)  
X_train_pca = pca.fit_transform(X_train)  
X_test_pca = pca.transform(X_test)  
  
# 绘制PCA降维后的三维图  
fig = plt.figure(figsize=(10, 8))  
ax = fig.add_subplot(111, projection='3d')  
ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], X_train_pca[:, 2], c=y_train, cmap='viridis', marker='o', label='Training data')  
ax.set_xlabel('PCA Feature 1')  
ax.set_ylabel('PCA Feature 2')  
ax.set_zlabel('PCA Feature 3')  
ax.set_title('3D visualization of Iris dataset after PCA')  
plt.legend()  
plt.show()  
  
# 使用SVM建模（不使用PCA）  
svm_no_pca = SVC(kernel='linear', C=1, random_state=42)  
svm_no_pca.fit(X_train, y_train)  
y_pred_no_pca = svm_no_pca.predict(X_test)  
accuracy_no_pca = accuracy_score(y_test, y_pred_no_pca)  
print(f'Accuracy without PCA: {accuracy_no_pca}')  
  
# 使用SVM建模（使用PCA）  
svm_pca = SVC(kernel='linear', C=1, random_state=42)  
svm_pca.fit(X_train_pca, y_train)  
y_pred_pca = svm_pca.predict(X_test_pca)  
accuracy_pca = accuracy_score(y_test, y_pred_pca)  
print(f'Accuracy with PCA: {accuracy_pca}')  
  
# 对比结果  
print(f'Comparison: {accuracy_no_pca} vs {accuracy_pca}')