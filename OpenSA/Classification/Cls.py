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
import numpy as np
import sklearn
from sklearn.calibration import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score,auc, classification_report,roc_curve,precision_recall_curve,f1_score
from lazypredict.Supervised import LazyClassifier
from sklearn.pipeline import Pipeline
import sklearn.svm
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier


# from sklearn.base import accuracy_score
# from sklearn.base import accuracy_score
from Classification.ClassicCls import ANN, SVM, PLS_DA, RF, SVM_with_PSO
from Classification.CNN import CNN
from Classification.SAE import SAE
from airpls import airPLS_deBase

def  QualitativeAnalysis(model, X_train, X_test, y_train, y_test,Featuresecletidx) -> float:

    if model == "PLS_DA":
        acc = PLS_DA(X_train, X_test, y_train, y_test)
    elif model == "ANN":
        acc = ANN(X_train, X_test, y_train, y_test,Featuresecletidx)
    elif model == "SVM":
        acc = SVM(X_train, X_test, y_train, y_test,Featuresecletidx)
    elif model == "SVM_with_PSO":
        acc = SVM_with_PSO(X_train, X_test, y_train, y_test, Featuresecletidx)
    elif model == "RF":
        acc = RF(X_train, X_test, y_train, y_test,Featuresecletidx)
    elif model == "CNN":
        acc = CNN(X_train, X_test, y_train, y_test, 16, 160, 4)
    elif model == "SAE":
        acc = SAE(X_train, X_test, y_train, y_test)
    elif model =="Lazy":
        clf = LazyClassifier()
        models_test,_what= clf.fit(X_train, X_test, y_train, y_test)
        print(models_test)
        print("????????????????????")
        print(_what)
        print("????????????????????")
        models_test, predictions_test = models_test,_what

        sns.set_theme(style='whitegrid')
        plt.figure(figsize=(5, 10))
        models_test['Accuracy (%)'] = models_test['Accuracy']*100
        ax = sns.barplot(y=models_test.index, x='Accuracy (%)', data=models_test)
        for i in ax.containers:
         ax.bar_label(i, fmt='%.2f')
        plt.show()
        # print("--------------model-----------")
        # www = clf.provide_models(X_train, X_test, y_train, y_test)
        # print(type(www))
        # with open("Dictionary_file.txt", "w") as file:
        #     # 遍历字典的键值对，并将其写入文件中
        #     for key, value in www.items():
        #         file.write(f"{key}: {value}\n")
        # print("--------------model-----------")

        # preprocessor = ColumnTransformer(
        # transformers=[
        # ('numeric', Pipeline([
        #     ('imputer', SimpleImputer()),
        #     ('scaler', StandardScaler())
        # ]), slice(0, 40)),  # 数值特征的处理
        # ('categorical_low', Pipeline([
        #     ('imputer', SimpleImputer(fill_value='missing', strategy='constant')),
        #     ('encoding', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        # ]), []),
        # ('categorical_high', Pipeline([
        #     ('imputer', SimpleImputer(fill_value='missing', strategy='constant')),
        #     ('encoding', OrdinalEncoder())
        # ]), [])
        # ]
        # )

        # # 定义SVC分类器
        # svm_classifier = SVC(random_state=42)

        # # 定义XGBoost分类器
        # xgb_classifier = XGBClassifier(
        # objective='multi:softprob',  # 多分类问题
        # random_state=42
        # )

        # # 定义完整的Pipeline，包括数据预处理和分类器
        # svm_pipeline = Pipeline(steps=[
        # ('preprocessor', preprocessor),
        # ('classifier', svm_classifier)
        # ])

        # svm_pipeline.fit(X_train, y_train)
        # svm_y_pred = svm_pipeline.predict(X_test)

        # print("SVM分类器性能报告:")
        # print(classification_report(y_test, svm_y_pred))
        # print("???????????????")

        # nowpath    = 'F://github//graduate-code//OpenSA//OpenSA' 
        # path =  nowpath+'//Data//Cls//test_output.csv'
        # Nirdata = np.loadtxt(open(path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        # x_newTest = Nirdata[:, :-1]
        # y_newTest = Nirdata[:, -1]

        # if(1):#关闭airpls
        #  for i in range(x_newTest.shape[0]):
        #      x_newTest[i] = airPLS_deBase(x_newTest[i])


        # if(1):
        #     min_value = np.min(x_newTest)  
        #     max_value = np.max(x_newTest)  

        # # 对整个数据集进行归一化  
        # normalized_data = (x_newTest - min_value) / (max_value - min_value)  
        # x_newTest = normalized_data



        # if(len(Featuresecletidx)!=0 ):
        #     print(Featuresecletidx)
        #     x_newTest = x_newTest[:,Featuresecletidx]
        #     print("不喜欢，",x_newTest.shape)    

        # y_new_pred = svm_pipeline.predict(x_newTest)
        # print("预测值:",y_new_pred)
        # print("真实值:",y_newTest)
        # newacc = accuracy_score(y_new_pred, y_newTest.ravel())
        # print("预测集的效果为：",newacc)



        # base_classifier = DecisionTreeClassifier()

        # # 初始化Bagging分类器
        # bagging_classifier = BaggingClassifier(base_estimator=base_classifier, n_estimators=10, random_state=42)

        # # 训练Bagging分类器
        # bagging_classifier.fit(X_train, y_train)

        # # 在测试集上进行预测
        # y_pred = bagging_classifier.predict(X_test)

        # # 计算准确率
        # accuracy = accuracy_score(y_test, y_pred)
        # print("Bagging Classifier Accuracy:", accuracy)
        
        # y_new_pred = bagging_classifier.predict(x_newTest)
        # print("预测值:",y_new_pred)
        # print("真实值:",y_newTest)
        # newacc = accuracy_score(y_new_pred, y_newTest.ravel())
        # print("预测集的效果为：",newacc)
        


        # 初始化基分类器（这里使用决策树作为基分类器）





     
        acc =1
        # print(x_newTest.shape)

    else:
        print("no this model of QuantitativeAnalysis")

    return acc