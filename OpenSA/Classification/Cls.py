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
import numpy as np
import sklearn
from sklearn.calibration import LinearSVC
from sklearn.metrics import accuracy_score,auc,roc_curve,precision_recall_curve,f1_score
# from lazypredict.Supervised import LazyClassifier

# from sklearn.base import accuracy_score
# from sklearn.base import accuracy_score
from Classification.ClassicCls import ANN, SVM, PLS_DA, RF
from Classification.CNN import CNN
from Classification.SAE import SAE

def  QualitativeAnalysis(model, X_train, X_test, y_train, y_test) -> float:

    if model == "PLS_DA":
        acc = PLS_DA(X_train, X_test, y_train, y_test)
    elif model == "ANN":
        acc = ANN(X_train, X_test, y_train, y_test)
    elif model == "SVM":
        acc = SVM(X_train, X_test, y_train, y_test)
    elif model == "RF":
        acc = RF(X_train, X_test, y_train, y_test)
    elif model == "CNN":
        acc = CNN(X_train, X_test, y_train, y_test, 16, 160, 4)
    elif model == "SAE":
        acc = SAE(X_train, X_test, y_train, y_test)
    elif model =="Lazy":
        acc = 1
        clf = LazyClassifier()
    
 

        models_test, predictions_test = clf.fit(X_train, X_test, y_train, y_test)
        print(models_test)
        best_model = models_test.index[0]
        print(f"The best performing model is: {best_model}")
        ccc= LinearSVC()
        ccc.fit(X_train, X_test)
        y_pred = ccc.predict(X_test)
        print(f"The accuracy of the LinearSVC model is: {accuracy_score(y_pred, y_test)}")

        # best_model = models.index(0)
        # print(f"The best performing model is: {best_model}")
        # _ = model.fit(X_train, y_train)
        # y_pred = model.predict(X_test)

        # label_map = {
        # 0: 'buluofen',
        # 1: 'duiyixiananjifen',
        # 2: 'fufangduiyixiananjifen',
        # 3: 'junmeishu',
        # 4: 'malaisuanlvnaming'
        # }
        # joblib.dump(label_map, 'label_map.pkl')
        # loaded_label_map = joblib.load('label_map.pkl')
        # nowpath    = 'F://github//graduate-code//OpenSA//OpenSA' 
        # path =  nowpath+'//new_test_output.csv'
        # Nirdata = np.loadtxt(open(path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
        # x_newTest = Nirdata[:, :-1]
        # y_new_pred = clf.predict(x_newTest)
        # print(y_new_pred)
        # y_pred_str = [loaded_label_map[label] for label in y_new_pred]
        # print(y_pred_str)

        # print(x_newTest.shape)

    else:
        print("no this model of QuantitativeAnalysis")

    return acc