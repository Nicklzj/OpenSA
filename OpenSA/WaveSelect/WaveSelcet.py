"""
    -*- coding: utf-8 -*-
    @Time   :2022/04/12 17:10
    @Author : Pengyou FU
    @blogs  : https://blog.csdn.net/Echo_Code?spm=1000.2115.3001.5343
    @github : https://github.com/FuSiry/OpenSA
    @WeChat : Fu_siry
    @License：Apache-2.0 license

"""

import os
from time import sleep
from WaveSelect.Lar import Lar
from WaveSelect.Spa import SPA
from WaveSelect.Uve import UVE
from WaveSelect.Cars import CARS_Cloud
from WaveSelect.Pca import Pca
from WaveSelect.GA import GA
from sklearn.model_selection import train_test_split

def SpctrumFeatureSelcet(method, X, y):
    """
       :param method: 波长筛选/降维的方法，包括：Cars, Lars, Uve, Spa, Pca
       :param X: 光谱数据, shape (n_samples, n_features)
       :param y: 光谱数据对应标签：格式：(n_samples，)
       :return: X_Feature： 波长筛选/降维后的数据, shape (n_samples, n_features)
                y：光谱数据对应的标签, (n_samples，)
    """
    Featuresecletidx=[]
    if method == "None":
        X_Feature = X
    elif method== "Cars":
        Featuresecletidx = CARS_Cloud(X, y)
        X_Feature = X[:, Featuresecletidx]
    elif method == "Lars":
        Featuresecletidx = Lar(X, y)
        X_Feature = X[:, Featuresecletidx]
    elif method == "Uve":
        Uve = UVE(X, y, 7)
        Uve.calcCriteria()
        Uve.evalCriteria(cv=5)
        Featuresecletidx = Uve.cutFeature(X)
        X_Feature = Featuresecletidx[0]
    elif method == "Spa":
        Xcal, Xval, ycal, yval = train_test_split(X, y, test_size=0.2)
        Featuresecletidx = SPA().spa(
            Xcal= Xcal, ycal=ycal, m_min=8, m_max=50, Xval=Xval, yval=yval, autoscaling=1)
        X_Feature = X[:, Featuresecletidx]
    elif method == "GA":
        file_path = "my_featurelist.txt"
        if os.path.exists(file_path):
            # 如果文件存在，则读取每一行的内容到列表中
            with open(file_path, "r") as file:
                feature_list = file.readlines()
                # 去除每一行末尾的换行符
                feature_list = [int(line.strip()) for line in feature_list]
            print("文件存在，内容已读取到列表中。")
            print("列表内容:", feature_list)
            Featuresecletidx = feature_list
        else:
            Featuresecletidx = GA(X, y, 10)
            print("遗传算法选出的特征点：",Featuresecletidx)
            with open('my_featurelist.txt', 'w') as f:
                for item in Featuresecletidx:
                    f.write(str(item) + '\n')
            # sleep(5)
        X_Feature = X[:, Featuresecletidx]
        
    elif method == "Pca":
        X_Feature = Pca(X)
    else:
        print("no this method of SpctrumFeatureSelcet!")

    return X_Feature, y,Featuresecletidx