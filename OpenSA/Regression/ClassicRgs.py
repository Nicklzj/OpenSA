
import os
import pickle

import joblib
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
# import hpelm

"""
    -*- coding: utf-8 -*-
    @Time   :2022/04/12 17:10
    @Author : Pengyou FU
    @blogs  : https://blog.csdn.net/Echo_Code?spm=1000.2115.3001.5343
    @github : https://github.com/FuSiry/OpenSA
    @WeChat : Fu_siry
    @License：Apache-2.0 license

"""

from sklearn.svm import SVR
from Evaluate.RgsEvaluate import ModelRgsevaluate
# from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoLarsCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense

def BpNeuralNetworkEvaluate(X_train, X_test, y_train, y_test):
    # 构建神经网络模型
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(units=1))

    # 编译模型
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 训练模型
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    # 预测并评估模型
    y_pred = model.predict(X_test)
    Rmse = mean_squared_error(y_test, y_pred, squared=False)
    R2 = r2_score(y_test, y_pred)
    Mae = mean_absolute_error(y_test, y_pred)

    return Rmse, R2, Mae

def D1_nnEvaluate(X_train, X_test, y_train, y_test):
    # 将输入数据转换成适用于 1D CNN 模型的形状
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # 构建 1D CNN 模型
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 训练 1D CNN 模型
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    # 预测并评估模型
    y_pred = model.predict(X_test)
    Rmse = mean_squared_error(y_test, y_pred, squared=False)
    R2 = r2_score(y_test, y_pred)
    Mae = mean_absolute_error(y_test, y_pred)

    return Rmse, R2, Mae


def LstmEvaluate(X_train, X_test, y_train, y_test):
    # 将输入数据转换成适用于 LSTM 模型的形状
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # 构建 LSTM 模型
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(1, X_train.shape[2])))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 训练 LSTM 模型
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    # 预测并评估模型
    y_pred = model.predict(X_test)
    Rmse = mean_squared_error(y_test, y_pred, squared=False)
    R2 = r2_score(y_test, y_pred)
    Mae = mean_absolute_error(y_test, y_pred)

    return Rmse, R2, Mae





def BayesianRidgeEvaluate(X_train, X_test, y_train, y_test):
   
    model_path = os.path.join(r"./", 'BayesianModel.pkl')
    if(model_path):
        model = joblib.load('BayesianModel.pkl')
        y_pred = model.predict(X_test)
    else:
        model = BayesianRidge()        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    # Evaluate the model
    Rmse = mean_squared_error(y_test, y_pred, squared=False)
    R2 = r2_score(y_test, y_pred)
    Mae = mean_absolute_error(y_test, y_pred)
    return Rmse, R2, Mae


def LassoLarsCVEvaluate(X_train, X_test, y_train, y_test):
    model = LassoLarsCV()
    # Fit the model
    model.fit(X_train, y_train)

    # Predict the values
    y_pred = model.predict(X_test)

    # Evaluate the model
    Rmse = mean_squared_error(y_test, y_pred, squared=False)
    R2 = r2_score(y_test, y_pred)
    Mae = mean_absolute_error(y_test, y_pred)

    return Rmse, R2, Mae


def Pls( X_train, X_test, y_train, y_test):


    model = PLSRegression(n_components=8)
    # fit the model
    model.fit(X_train, y_train)

    # predict the values
    y_pred = model.predict(X_test)

    Rmse, R2, Mae = ModelRgsevaluate(y_pred, y_test)

    return Rmse, R2, Mae


from lazypredict.Supervised import LazyRegressor

def Lazy( X_train, X_test, y_train, y_test):

    reg = LazyRegressor(ignore_warnings=False, custom_metric=None)
    # Fit LazyRegressor
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    # Print the models
    print(models)

    import sys

    # 保存原来的 sys.stdout
    original_stdout = sys.stdout
    with open('regressor_output.txt', 'w') as f:
        # 重定向 sys.stdout 到文件
   
        sys.stdout = f 
        print(models)
 
    # 恢复原来的 sys.stdout
    sys.stdout = original_stdout


    
    # best_model = models.iloc[0]
    # print("----------------------------------------")
    # print(best_model)
    # print("----------------------------------------")
    # print(predictions)
    return None, None, None  # 或者根据你的需要进行处理

def Svregression(X_train, X_test, y_train, y_test):


    model = SVR(C=2, gamma=1e-07, kernel='linear')
    # model.fit(X_train, y_train)
    # 创建多输出回归器
    model = MultiOutputRegressor(model)
    # 拟合模型
    model.fit(X_train, y_train)

    # predict the values
    y_pred = model.predict(X_test)
    # Rmse, R2, Mae = ModelRgsevaluate(y_pred, y_test)
    # 评估模型
    mse = mean_squared_error(y_test, y_pred)
    return mse, 0, 0

def Anngression(X_train, X_test, y_train, y_test):


    model = MLPRegressor(
        hidden_layer_sizes=(20, 20), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
        learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=400, shuffle=True,
        random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.fit(X_train, y_train)

    # predict the values
    y_pred = model.predict(X_test)
    Rmse, R2, Mae = ModelRgsevaluate(y_pred, y_test)

    return Rmse, R2, Mae

def ELM(X_train, X_test, y_train, y_test):

    model = hpelm.ELM(X_train.shape[1], 1)
    model.add_neurons(20, 'sigm')


    model.train(X_train, y_train, 'r')
    y_pred = model.predict(X_test)


    Rmse, R2, Mae = ModelRgsevaluate(y_pred, y_test)

    return Rmse, R2, Mae