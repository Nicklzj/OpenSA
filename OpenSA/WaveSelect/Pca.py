"""
    -*- coding: utf-8 -*-
    @Time   :2022/04/12 17:10
    @Author : Pengyou FU
    @blogs  : https://blog.csdn.net/Echo_Code?spm=1000.2115.3001.5343
    @github : https://github.com/FuSiry/OpenSA
    @WeChat : Fu_siry
    @License：Apache-2.0 license

"""

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

def Pca(X, nums=3):
    """
       :param X: raw spectrum data, shape (n_samples, n_features)
       :param nums: Number of principal components retained
       :return: X_reduction：Spectral data after dimensionality reduction
    """
    pca = PCA(n_components=3)  # 保留的特征数码
    pca.fit(X)
    X_reduction = pca.transform(X)


    fig = plt.figure()  
    ax = fig.add_subplot(111, projection='3d')  

    # 绘制数据点  
    ax.scatter(X_reduction[:, 0], X_reduction[:, 1], X_reduction[:, 2])  

    # 设置坐标轴标签  
    ax.set_xlabel('PC1')  
    ax.set_ylabel('PC2')  
    ax.set_zlabel('PC3')  

    # 显示图形  
    plt.show()


    return X_reduction
