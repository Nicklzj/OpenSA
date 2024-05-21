import argparse
import os
import pathlib
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
# from pyswarm import pso
from airpls import airPLS_deBase

parser = argparse.ArgumentParser(description='Your Description Here')

# 添加参数选项
parser.add_argument('--filename', type=str, default='None',
                        help='Description of filename option.')
parser.add_argument('--modelname', type=str, default='pso_random_forest_model.pkl',
                        help='Description of model option.')
args = parser.parse_args()


with open(args.modelname, 'rb') as file:
    RF = pickle.load(file)
print("Model loaded from file.")


import csv

# 定义存储x和y值的列表
x_list = []
y_list = []

# 读取CSV文件
# file_path = 'liuweny.csv'  # 请将your_file.csv替换为你的CSV文件路径
#filename = '/home/rock/liuweny.csv'
print(args.filename)
with open(args.filename, mode='r', newline='') as csvfile:
    csvreader = csv.reader((csvfile))
    
    # 逐行读取数据
    for row in csvreader:
        x_list.append(float(row[0]))
        y_list.append(float(row[1]))

# 打印读取的数据
print("x_list:", x_list)
print("y_list:", y_list)




x_newTest = []
x_newTest.append(y_list[:-1])

x_newTest = np.array(x_newTest)

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

y_new_pred = RF.predict(x_newTest)
print("预测值：",y_new_pred)
