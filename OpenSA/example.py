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
from matplotlib import colors
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
# from lazypredict.Supervised import LazyClassifier
# from lazypredict.Supervised import LazyRegressor
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
def SpectralQualitativeAnalysis(data, label, ProcessMethods, FslecetedMethods, SetSplitMethods, model,Featuresecletidx):

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
    FeatrueData, labels,Featuresecletidx = SpctrumFeatureSelcet(FslecetedMethods, ProcesedData, label)
    print( "波段选择成功,特征数据的形状为：",FeatrueData.shape)


    X_train, X_test, y_train, y_test = SetSplit(SetSplitMethods, FeatrueData, labels, test_size=0.15, randomseed=100)
 
    acc = QualitativeAnalysis(model, X_train, X_test, y_train, y_test,Featuresecletidx )
    
    return acc




 

if __name__ == '__main__':
    mode ="dingxing"
    if(mode=="dingliang"):
        data2, label2 = LoadNirtest('Rgs')
        RMSE, R2, MAE = SpectralQuantitativeAnalysis(data2, label2, "SNV", "Uve", "ks", "Pls")
        print("The Pca RMSE:{} R2:{}, MAE:{} of result!".format(RMSE, R2, MAE))
    
    else:
        data1, label1 = LoadNirtest('Cls')

    for i in range(data1.shape[0]):
            data1[i] = airpls.airPLS_deBase(data1[i])
    print("-----------去基线成功----------------")
            # 计算最大值和最小值  
    min_value = np.min(data1)  
    max_value = np.max(data1)  
        
    # 对整个数据集进行归一化  
    normalized_data = (data1 - min_value) / (max_value - min_value)  
    data1 = normalized_data

    acc = SpectralQualitativeAnalysis(data1, label1, "None", "None", "ks", "RF",[])
    print("The model  acc:{} of result!".format(acc))

