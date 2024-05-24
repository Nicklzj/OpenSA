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
        if(len(row)==0):
            break;
        x_list.append(float(row[0]))
        y_list.append(float(row[1]))

# 打印读取的数据
print("x_list:", x_list)
print("y_list:", y_list)

std_filename = '/home/rock/liuweny.csv'
std_x_list = []
std_y_list = []
with open(std_filename, mode='r', newline='') as csvfile:
    csvreader = csv.reader((csvfile))
    
    # 逐行读取数据
    for row in csvreader:
        if(len(row)==0):
            break;
        std_x_list.append(float(row[0]))
        std_y_list.append(float(row[1]))

# 打印读取的数据
# print("x_list:", std_x_list)
# print("y_list:", std_y_list)


Comp1=np.array(y_list)
Comp2=np.array(std_y_list)
cosine_similarity = np.dot(Comp1, Comp2) / (np.linalg.norm(Comp1) * np.linalg.norm(Comp2))
pearson_corr = np.corrcoef(Comp1, Comp2)[0, 1]


# from scipy.spatial.distance import euclidean
# from fastdtw import fastdtw

# distance, path = fastdtw(Comp1, Comp2, dist=euclidean)
# print(f"DTW Distance: {distance}")


# import numpy as np
# from scipy.fftpack import fft

# # 傅里叶变换
# fft_X = np.abs(fft(Comp1))
# fft_Y = np.abs(fft(Comp2))

# # 计算频域特征的相似性
# freq_similarity = np.dot(fft_X, fft_Y) / (np.linalg.norm(fft_X) * np.linalg.norm(fft_Y))
# print(f"Frequency Domain Cosine Similarity: {freq_similarity}")

# from sklearn.metrics import mutual_info_score

# mutual_info = mutual_info_score(Comp1, Comp2)
# print(f"Mutual Information: {mutual_info}")


# 均方误差
mse = np.mean((Comp1 - Comp2) ** 2)

# 欧氏距离
euclidean_distance = np.linalg.norm(Comp1 - Comp2)

print(f"Pearson Correlation: {pearson_corr}")
print(f"Cosine Similarity: {cosine_similarity}")
print(f"Mean Squared Error: {mse}")
print(f"Euclidean Distance: {euclidean_distance}")

 





y_new_pred=[]
y_new_pred.append(999)
if(pearson_corr>=0.80 and pearson_corr >=0.80):
	x_newTest = []
	x_newTest.append(y_list[:-1])

	x_newTest = np.array(x_newTest)

	if(1):#    
		for i in range(x_newTest.shape[0]):
			x_newTest[i] = airPLS_deBase(x_newTest[i])
	if(1):
	    min_value = np.min(x_newTest)  
	    max_value = np.max(x_newTest)    
	    # 对整个数据集进行归一化  
	    normalized_data = (x_newTest - min_value) / (max_value - min_value)  
	    x_newTest = normalized_data
        
	y_new_pred = RF.predict(x_newTest)
    # RF.predict_proba(x_newTest)
    # probabilities=RF.predict_proba(x_newTest)
    # #probabilities=RF.predict_proba(x_newTest)
    # #first_sample_prob = probabilities[0]

    # # print(f"类别0的概率: {first_sample_prob[0]}")
    # # print(f"类别1的概率: {first_sample_prob[1]}")
    # # print(f"类别2的概率: {first_sample_prob[2]}")
    # # 获取每棵树对测试样本的预测结果
    # tree_predictions = np.array([tree.predict(x_newTest) for tree in RF.estimators_])

 






beverages = {
    0: "BiLiTong_Origin",
    2: "DongBei_Imitation",
    3:"WeiFuJia_Imitation",
    4:"TongDe_Imitation",
    999:"UnKnown"
}
if(y_new_pred[0]==0 and (pearson_corr<0.99 or cosine_similarity<0.99)):
    y_new_pred[0]=2

if(y_new_pred[0]==2 and (pearson_corr>=0.99 or cosine_similarity>=0.99)):
    y_new_pred[0]=0



print("预测值：",beverages[y_new_pred[0]])
if(y_new_pred[0]==0):
    cosine_similarity=1
os.system("python3 Show_box.py --drugname %s --indicators %f" % (beverages[y_new_pred[0]],cosine_similarity))

