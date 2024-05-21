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
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score,auc, classification_report,roc_curve,precision_recall_curve,f1_score
# from lazypredict.Supervised import LazyClassifier
from sklearn.pipeline import Pipeline
import sklearn.svm
from sklearn.tree import DecisionTreeClassifier

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
from Classification.ClassicCls import ANN, SVM, PLS_DA, RF
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
    elif model == "RF":
        acc = RF(X_train, X_test, y_train, y_test,Featuresecletidx)
    elif model == "CNN":
        acc = CNN(X_train, X_test, y_train, y_test, 16, 160, 4)
    elif model == "SAE":
        acc = SAE(X_train, X_test, y_train, y_test)

    else:
        print("no this model of QuantitativeAnalysis")

    return acc