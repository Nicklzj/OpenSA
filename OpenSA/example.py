"""
    -*- coding: utf-8 -*-
    @Time   :2022/04/12 17:10
    @Author : Pengyou FU
    @blogs  : https://blog.csdn.net/Echo_Code?spm=1000.2115.3001.5343
    @github : https://github.com/FuSiry/OpenSA
    @WeChat : Fu_siry
    @License：Apache-2.0 license

"""


from time import sleep
import numpy as np
from sklearn.model_selection import train_test_split
from DataLoad.DataLoad import SetSplit, LoadNirtest
from Preprocessing.Preprocessing import Preprocessing
from WaveSelect.WaveSelcet import SpctrumFeatureSelcet
# from Plot.SpectrumPlot import plotspc
# from Plot.SpectrumPlot import ClusterPlot
from Simcalculation.SimCa import Simcalculation
from Clustering.Cluster import Cluster
from Regression.Rgs import QuantitativeAnalysis
from Classification.Cls import QualitativeAnalysis
from lazypredict.Supervised import LazyClassifier
from lazypredict.Supervised import LazyRegressor
import airpls
import matplotlib.pyplot as plt#导入强大的绘图库

#光谱聚类分析
def SpectralClusterAnalysis(data, label, ProcessMethods, FslecetedMethods, ClusterMethods):
    """
     :param data: shape (n_samples, n_features), 光谱数据
     :param label: shape (n_samples, ), 光谱数据对应的标签(理化性质)
     :param ProcessMethods: string, 预处理的方法, 具体可以看预处理模块
     :param FslecetedMethods: string, 光谱波长筛选的方法, 提供UVE、SPA、Lars、Cars、Pca
     :param ClusterMethods : string, 聚类的方法，提供Kmeans聚类、FCM聚类
     :return: Clusterlabels: 返回的隶属矩阵

     """
    ProcesedData = Preprocessing(ProcessMethods, data)
    FeatrueData, _ = SpctrumFeatureSelcet(FslecetedMethods, ProcesedData, label)
    Clusterlabels = Cluster(ClusterMethods, FeatrueData)
    #ClusterPlot(data, Clusterlabels)
    return Clusterlabels

# 光谱定量分析
def SpectralQuantitativeAnalysis(data, label, ProcessMethods, FslecetedMethods, SetSplitMethods, model):

    """
    :param data: shape (n_samples, n_features), 光谱数据
    :param label: shape (n_samples, ), 光谱数据对应的标签(理化性质)
    :param ProcessMethods: string, 预处理的方法, 具体可以看预处理模块
    :param FslecetedMethods: string, 光谱波长筛选的方法, 提供UVE、SPA、Lars、Cars、Pca
    :param SetSplitMethods : string, 划分数据集的方法, 提供随机划分、KS划分、SPXY划分
    :param model : string, 定量分析模型, 包括ANN、PLS、SVR、ELM、CNN、SAE等，后续会不断补充完整
    :return: Rmse: float, Rmse回归误差评估指标
             R2: float, 回归拟合,
             Mae: float, Mae回归误差评估指标
    """
    # 要看效果好不好，如果效果不好，就不去基线了
    # for i in range(data.shape[0]):
    #     data[i] = airpls.airPLS_deBase(data[i])
    # print("-----------去基线成功----------------")
    ProcesedData = Preprocessing(ProcessMethods, data)
    FeatrueData, labels = SpctrumFeatureSelcet(FslecetedMethods, ProcesedData, label)
    # print("---------------------------------------")
    # print(FeatrueData.shape)
    # print(labels.shape)
    # print("---------------------------------------")

    # sleep(100)
    # 将特征值从很多很多降到了固定的 x个               def Pca(X, nums=20):
    X_train, X_test, y_train, y_test = SetSplit(SetSplitMethods, FeatrueData, labels, test_size=0.2, randomseed=123)
    # reg = LazyRegressor(ignore_warnings=False, custom_metric=None)
    # models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    # print(models)
    # best_model = models.iloc[0]
    Rmse, R2, Mae = QuantitativeAnalysis(model, X_train, X_test, y_train, y_test ) 
    # model.save('model_savedmodel', save_format='tf')  
    return Rmse, R2, Mae

# 光谱定性分析
def SpectralQualitativeAnalysis(data, label, ProcessMethods, FslecetedMethods, SetSplitMethods, model):

    """
    :param data: shape (n_samples, n_features), 光谱数据
    :param label: shape (n_samples, ), 光谱数据对应的标签(理化性质)
    :param ProcessMethods: string, 预处理的方法, 具体可以看预处理模块
    :param FslecetedMethods: string, 光谱波长筛选的方法, 提供UVE、SPA、Lars、Cars、Pca
    :param SetSplitMethods : string, 划分数据集的方法, 提供随机划分、KS划分、SPXY划分
    :param model : string, 定性分析模型, 包括ANN、PLS_DA、SVM、RF、CNN、SAE等，后续会不断补充完整
    :return: acc： float, 分类准确率
    """

    ProcesedData = Preprocessing(ProcessMethods, data)
    FeatrueData, labels = SpctrumFeatureSelcet(FslecetedMethods, ProcesedData, label)
    X_train, X_test, y_train, y_test = SetSplit(SetSplitMethods, FeatrueData, labels, test_size=0.2, randomseed=123)
    acc = QualitativeAnalysis(model, X_train, X_test, y_train, y_test )
    # model.save('model_savedmodel', save_format='tf')  
    
    return acc









if __name__ == '__main__':

    # ## 载入原始数据并可视化
    # data1, label1 = LoadNirtest('Cls')
    # #plotspc(data1, "raw specturm")
    # # 光谱定性分析演示
    # # 示意1: 预处理算法:MSC , 波长筛选算法: 不使用, 全波长建模, 数据集划分:随机划分, 定性分析模型: RF
    # acc = SpectralQualitativeAnalysis(data1, label1, "MSC", "Lars", "random", "PLS_DA")
    # print("The acc:{} of result!".format(acc))
    mode ="dingliang"
    if(mode=="dingliang"):
    ## 载入原始数据并可视化
    # 光谱定量分析演示
    # 示意1: 预处理算法:MSC , 波长筛选算法: Uve, 数据集划分:KS, 定性分量模型: SVR
        data2, label2 = LoadNirtest('Rgs')
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")  
        print(data2.shape)
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        print("yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
        print(label2.shape) 
        print("yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
        RMSE, R2, MAE = SpectralQuantitativeAnalysis(data2, label2, "SG", "None", "ks", "SVR")
        print("The Pca RMSE:{} R2:{}, MAE:{} of result!".format(RMSE, R2, MAE))
    
    else:
        data1, label1 = LoadNirtest('Cls')
        acc = SpectralQualitativeAnalysis(data1, label1, "MSC", "Lars", "random", "RF")
        print("The acc:{} of result!".format(acc))

