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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import sklearn.svm as svm
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
import pandas  as pd
import pickle

# from OpenSA.Classification.DataLoad import LoadNirtest

 
def ANN(X_train, X_test, y_train, y_test, StandScaler=None):

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




    #添加保存模型的代码
 
    # print (clf2.predict(X[0:1]))
    # joblib.dump(clf, 'data.pkl')#也可以使用文件对象
    # clf = joblib.load('data.pkl') 
    print("-------------------------\n")
    print(clf)
    print("-------------------------\n")
    return acc

def SVM(X_train, X_test, y_train, y_test):

    clf = svm.SVC(C=1, gamma=1e-3)
    clf.fit(X_train, y_train)

    predict_results = clf.predict(X_test)
    acc = accuracy_score(predict_results, y_test.ravel())


    label_map = {
        0: 'buluofen',
        1: 'duiyixiananjifen',
        2: 'fufangduiyixiananjifen',
        3: 'junmeishu',
        4: 'malaisuanlvnaming'
    }
    joblib.dump(clf, 'model.pkl')
    joblib.dump(label_map, 'label_map.pkl')

    # 加载模型和标签映射
    loaded_model = joblib.load('model.pkl')
    loaded_label_map = joblib.load('label_map.pkl')


    nowpath    = 'F://github//graduate-code//OpenSA' 
    path =  nowpath+'//new_test.csv'

    Nirdata = np.loadtxt(open(path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
    
    x_newTest = Nirdata[:, :-1]
    y_new_pred = loaded_model.predict(x_newTest)
    print(y_new_pred)
    y_pred_str = [loaded_label_map[label] for label in y_new_pred]
    print(y_pred_str)

  
    print(x_newTest.shape)


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


    label_map = {
        0: 'buluofen',
        1: 'duiyixiananjifen',
        2: 'fufangduiyixiananjifen',
        3: 'junmeishu',
        4: 'malaisuanlvnaming'
    }
    joblib.dump(model, 'model.pkl')
    joblib.dump(label_map, 'label_map.pkl')

    # 加载模型和标签映射
    loaded_model = joblib.load('model.pkl')
    loaded_label_map = joblib.load('label_map.pkl')


    nowpath    = 'F://github//graduate-code//OpenSA' 
    path =  nowpath+'//new_test.csv'

    Nirdata = np.loadtxt(open(path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
    
    x_newTest = Nirdata[:, :-1]
    y_new_pred = loaded_model.predict(x_newTest)
    print(x_newTest)
    print(y_new_pred)
    y_pred_str = [loaded_label_map[labelS] for labelS in y_new_pred]
    # print(y_pred_str)
    # print(x_newTest.shape)



    return acc

def RF(X_train, X_test, y_train, y_test):

    RF = RandomForestClassifier(n_estimators=15,max_depth=3,min_samples_split=3,min_samples_leaf=3)
    RF.fit(X_train, y_train)

    y_pred = RF.predict(X_test)    
    acc = accuracy_score(y_test, y_pred)
 

    label_map = {
        0: 'buluofen',
        1: 'duiyixiananjifen',
        2: 'fufangduiyixiananjifen',
        3: 'junmeishu',
        4: 'malaisuanlvnaming'
    }
    joblib.dump(RF, 'model.pkl')
    joblib.dump(label_map, 'label_map.pkl')

    # 加载模型和标签映射
    loaded_model = joblib.load('model.pkl')
    loaded_label_map = joblib.load('label_map.pkl')


    nowpath    = 'F://github//graduate-code//OpenSA' 
    path =  nowpath+'//new_test.csv'

    Nirdata = np.loadtxt(open(path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
    
    x_newTest = Nirdata[:, :-1]
    y_new_pred = loaded_model.predict(x_newTest)
    print(y_new_pred)
    y_pred_str = [loaded_label_map[label] for label in y_new_pred]
    print(y_pred_str)

  
    print(x_newTest.shape)

    return acc
