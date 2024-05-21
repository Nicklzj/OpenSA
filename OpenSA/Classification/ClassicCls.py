"""
    -*- coding: utf-8 -*-
    @Time   :2022/04/12 17:10
    @Author : Pengyou FU
    @blogs  : https://blog.csdn.net/Echo_Code?spm=1000.2115.3001.5343
    @github : https://github.com/FuSiry/OpenSA
    @WeChat : Fu_siry
    @License：Apache-2.0 license

"""


import joblib
from matplotlib import pyplot as plt
from pyparsing import nums
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import sklearn.svm as svm
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
import pandas  as pd
import pickle
import os
from airpls import airPLS_deBase

# from OpenSA.Classification.DataLoad import LoadNirtest

 
def ANN(X_train, X_test, y_train, y_test, Featuresecletidx,StandScaler=None):

    if StandScaler:
        scaler = StandardScaler() # 标准化转换
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # 神经网络输入为2，第一隐藏层神经元个数为5，第二隐藏层神经元个数为2，输出结果为2分类。
    # solver='lbfgs',  MLP的求解方法：L-BFGS 在小数据上表现较好，Adam 较为鲁棒，
    # SGD在参数调整较优时会有最佳表现（分类效果与迭代次数）,SGD标识随机梯度下降。
    #clf =  MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(8,8), random_state=1, activation='relu')

    # clf =  MLPClassifier()
    # joblib.dump(clf, 'data.pkl')#也可以使用文件对象

    clf =  MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9,
                  beta_2=0.999, early_stopping=False, epsilon=1e-08,
                  hidden_layer_sizes=(10, 8), learning_rate='constant',
                  learning_rate_init=0.001, max_iter=200, momentum=0.9,
                  nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                  solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
                  warm_start=False)

    clf.fit(X_train,y_train.ravel())
    predict_results=clf.predict(X_test)
    acc = accuracy_score(predict_results, y_test.ravel())
    print("训练预测值：",predict_results)
    print("训练真实值：",y_test)
 



    path =   './/Data//Cls//test_output.csv'
    Nirdata = np.loadtxt(open(path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
    x_newTest = Nirdata[:, :-1]
    y_newTest = Nirdata[:, -1]


    min_value = np.min(x_newTest)  
    max_value = np.max(x_newTest)  
    # global minmaxFlag
    global airplsFlag
    if(1):#关闭airpls
        for i in range(x_newTest.shape[0]):
            x_newTest[i] = airPLS_deBase(x_newTest[i])
        print("测试集数据airpls已处理")
    
    if(1):
    # 对整个数据集进行归一化  
        normalized_data = (x_newTest - min_value) / (max_value - min_value)  
        x_newTest = normalized_data


   
    if(len(Featuresecletidx)!=0 ):
        print(Featuresecletidx)
        x_newTest = x_newTest[:,Featuresecletidx]
        print("不喜欢，",x_newTest.shape)    




    y_new_pred = clf.predict(x_newTest)
    print("预测值：",y_new_pred)
    print("真实值：",y_newTest)
    newacc = accuracy_score(y_new_pred, y_newTest.ravel())
    print("预测集的效果为：",newacc)

   
    return acc

def SVM(X_train, X_test, y_train, y_test,Featuresecletidx):

    clf = svm.SVC(C=2, gamma=1e-3)
    clf.fit(X_train, y_train)

    predict_results = clf.predict(X_test)
    print("x:",X_test.shape)
    acc = accuracy_score(predict_results, y_test.ravel())


    nowpath    = '.' 
    path =  nowpath+'//Data//Cls//test_output.csv'
    Nirdata = np.loadtxt(open(path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
    x_newTest = Nirdata[:, :-1]
    y_newTest = Nirdata[:, -1]

    if(1):#关闭airpls
        for i in range(x_newTest.shape[0]):
            x_newTest[i] = airPLS_deBase(x_newTest[i])

        
    if(1):
        min_value = np.min(x_newTest)  
        max_value = np.max(x_newTest)  
        
        # 对整个数据集进行归一化  
        normalized_data = (x_newTest - min_value) / (max_value - min_value)  
        x_newTest = normalized_data



    if(len(Featuresecletidx)!=0 ):
        print(Featuresecletidx)
        x_newTest = x_newTest[:,Featuresecletidx]
        print("不喜欢，",x_newTest.shape)    

    y_new_pred = clf.predict(x_newTest)
    print("预测值和真实值：",y_new_pred,y_newTest)
    newacc = accuracy_score(y_new_pred, y_newTest.ravel())
    print("预测集的效果为：",newacc)


    return acc

def PLS_DA(X_train, X_test, y_train, y_test):

    y_train = pd.get_dummies(y_train)
    # 建模
    model = PLSRegression(n_components=228)
    model.fit(X_train, y_train)
    # 预测
    y_pred = model.predict(X_test)
    # 将预测结果（类别矩阵）转换为数值标签
    y_pred = np.array([np.argmax(i) for i in y_pred])
    acc = accuracy_score(y_test, y_pred)
    print("预测值和真实值：")
    print(y_test, y_pred)

    # nowpath    = 'F://github//graduate-code//OpenSA//OpenSA//Data//Cls' 
    # path =  nowpath+'//test_output.csv'
    # Nirdata = np.loadtxt(open(path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
    # print("Nirdata.shape: ",Nirdata.shape)
    # x_newTest = Nirdata[:, :-1]
    # y_newTest = Nirdata[:,-1]

    # if(1):#    
    #     for i in range(x_newTest.shape[0]):
    #         x_newTest[i] = airPLS_deBase(x_newTest[i])
    #     print("对测试集已经做出了airpls")
    # if(1):
    #     min_value = np.min(x_newTest)  
    #     max_value = np.max(x_newTest)  
        
    #     # 对整个数据集进行归一化  
    #     normalized_data = (x_newTest - min_value) / (max_value - min_value)  
    #     x_newTest = normalized_data


    # x_newTest = x_newTest[-6:-1]
    # y_newTest = Nirdata[-6:-1,-1]
    # y_new_pred = model.predict(x_newTest)
    # y_new_pred = np.array([np.argmax(i) for i in y_pred])
    # print("预测值和真实值：",y_new_pred,y_newTest)
  



    return acc

def RF(X_train, X_test, y_train, y_test,Featuresecletidx):

    model_filename = 'pso_random_forest_model.pkl'

    # 检查模型文件是否存在
    if os.path.exists(model_filename):
        # 载入模型
        with open(model_filename, 'rb') as file:
            RF = pickle.load(file)
        print("Model loaded from file.")
    else:
        RF = RandomForestClassifier(n_estimators=500)
        RF.fit(X_train, y_train)
        with open('pso_random_forest_model.pkl', 'wb') as file:
            pickle.dump(RF, file)
 

    y_pred = RF.predict(X_test)   
    # print("真实值和预测值：",y_test, y_pred) 
    acc = accuracy_score(y_test, y_pred)


    nowpath    = '.' 
    path =  nowpath+'//Data//Cls//test_output.csv'
    Nirdata = np.loadtxt(open(path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
    x_newTest = Nirdata[:, :-1]
    y_newTest = Nirdata[:, -1]
    if(1):#    
        for i in range(x_newTest.shape[0]):
            x_newTest[i] = airPLS_deBase(x_newTest[i])
        print("对测试集已经做出了airpls")
    if(1):
        min_value = np.min(x_newTest)  
        max_value = np.max(x_newTest)    
        # 对整个数据集进行归一化  
        normalized_data = (x_newTest - min_value) / (max_value - min_value)  
        x_newTest = normalized_data
    if(len(Featuresecletidx)!=0 ):
        print(Featuresecletidx)      
        x_newTest = x_newTest[:,Featuresecletidx]
    print("不喜欢，",x_newTest.shape)
    y_new_pred = RF.predict(x_newTest)
    print("预测值和真实值：",y_new_pred,y_newTest)
    newacc = accuracy_score(y_new_pred, y_newTest.ravel())
    print("预测集的效果为：",newacc)

    for elem in enumerate(y_new_pred):
        if(elem[1] ==0):
            print(elem)
        # print(elem)
    return acc
