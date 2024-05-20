"""
    -*- coding: utf-8 -*-
    @Time   :2022/04/12 17:10
    @Author : Pengyou FU
    @blogs  : https://blog.csdn.net/Echo_Code?spm=1000.2115.3001.5343
    @github : https://github.com/FuSiry/OpenSA
    @WeChat : Fu_siry
    @License：Apache-2.0 license

"""

from Regression.ClassicRgs import BayesianRidgeEvaluate, BpNeuralNetworkEvaluate, D1_nnEvaluate, LassoLarsCVEvaluate, Lazy, Anngression, LstmEvaluate, Svregression, ELM,Pls
from Regression.CNN import CNNTrain

def  QuantitativeAnalysis(model, X_train, X_test, y_train, y_test):

    if model == "Lazy":
        Rmse, R2, Mae = Lazy(X_train, X_test, y_train, y_test)
    elif model == "BpNeuralNetworkEvaluate":
        Rmse, R2, Mae = BpNeuralNetworkEvaluate(X_train, X_test, y_train, y_test)
    elif model =="D1_nnEvaluate":
        Rmse, R2, Mae = D1_nnEvaluate(X_train, X_test, y_train, y_test)
    elif model == "LassoLarsCVEvaluate":
        Rmse, R2, Mae = LassoLarsCVEvaluate(X_train, X_test, y_train, y_test)
    elif model == "BayesianRidgeEvaluate":
        Rmse, R2, Mae = BayesianRidgeEvaluate(X_train, X_test, y_train, y_test)
    elif model =="Pls":
        Rmse, R2, Mae = Pls(X_train, X_test, y_train, y_test)
    elif model == "ANN":
        Rmse, R2, Mae = Anngression(X_train, X_test, y_train, y_test)
    elif model == "SVR":
        Rmse, R2, Mae = Svregression(X_train, X_test, y_train, y_test)
    elif model == "ELM":
        Rmse, R2, Mae = ELM(X_train, X_test, y_train, y_test)
    elif model == "CNN":
        # Rmse, R2, Mae = CNNTrain("AlexNet",X_train, X_test, y_train, y_test,  50)
        # Rmse, R2, Mae = CNNTrain("DeepSpectra",X_train, X_test, y_train, y_test,  50)
        Rmse, R2, Mae = CNNTrain("ConNet",X_train, X_test, y_train, y_test,  50)
    else:
        print("no this model of QuantitativeAnalysis")
    return Rmse, R2, Mae 