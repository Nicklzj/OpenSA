from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pyswarm import pso
import airpls
import kshuafen
#使用cnn进行了特征提取 svm用于建模分析

path = './/Data//Cls//classtable.csv'
Nirdata = np.loadtxt(open(path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
data1 = Nirdata[:, :-1]
label1 = Nirdata[:, -1]

list_of_lists =  []
for i in range(len(data1)):
    if(i==0):
        list_of_lists.append(data1[i])
    elif(label1[i]!=label1[i-1]):
        list_of_lists.append(data1[i])
colors = ['r', 'b','y','black']  # 使用不同的颜色
labellls = ['Origin 1-BiLiTong',  'Imitation 1:WeFuiry','Imitation 2:EastNorth','Imitation 3:TongDe']  # 每个数据集的标签
for sublist, color, labelll in zip(list_of_lists, colors, labellls):
    plt.plot(sublist, color=color, label=labelll)

plt.legend()
plt.show()


if(1):
    for i in range(data1.shape[0]):
            data1[i] = airpls.airPLS_deBase(data1[i])
    print("-----------去基线成功----------------")

            # 计算最大值和最小值  
    min_value = np.min(data1)  
    max_value = np.max(data1)  
        
    # 对整个数据集进行归一化  
    normalized_data = (data1 - min_value) / (max_value - min_value)  
    data1 = normalized_data


list_of_lists =  []
for i in range(len(label1)):
    if(i==0):
        list_of_lists.append(data1[i])
    elif(label1[i]!=label1[i-1]):
        list_of_lists.append(data1[i])
colors = ['r', 'b','y','black']  # 使用不同的颜色
labels = ['Origin1-BiLiTong',  'Imitation 1:WeFuiry','Imitation 2:EastNorth','Imitation 3:TongDe']  # 每个数据集的标签
for sublist, color, labell in zip(list_of_lists, colors, labels):
    plt.plot(sublist, color=color, label=labell)
plt.legend()
plt.show()


# 假设 x_data 是形状为 (num_samples, 2048) 的光谱数据，y_data 是相应的类别标签
x_data = data1
y_data = label1






# 转换为 PyTorch 张量
x_data = torch.tensor(x_data, dtype=torch.float32)
y_data = torch.tensor(y_data, dtype=torch.long)

# 将数据分成训练集和测试集
dataset = TensorDataset(x_data, y_data)
train_size = int(0.80 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


# kshuafen(x_data,train_size)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 构建CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(128 * 256, 128)
        # self.fc1 = nn.Linear(32768, 128)
        self.fc1 = nn.Linear(32640, 128)

        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        # self.fc2 = nn.Linear(128, 5)
        # self.fc1 = nn.Linear(32768, 1024)
        # self.fc2 = nn.Linear(1024, 128)
        # self.fc3 = nn.Linear(128, 32)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class No_pool_CNN(nn.Module):
    def __init__(self):
        super(No_pool_CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(65408, 128)
        # self.fc2 = nn.Linear(128, 64)
        # self.fc3 = nn.Linear(64, 16)
        self.fc1 = nn.Linear(65408, 4096)
        self.fc2 = nn.Linear(4096, 512)
        self.fc3 = nn.Linear(512, 16)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# 创建模型实例
cnn_model = CNN()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# 训练模型
num_epochs = 50
cnn_model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.unsqueeze(1)  # 添加一个维度以匹配输入形状 (batch_size, 1, 2048)
        
        # 零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = cnn_model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

# 提取特征
cnn_model.eval()
train_features = []
train_labels = []
with torch.no_grad():
    for inputs, labels in train_loader:
        inputs = inputs.unsqueeze(1)
        features = cnn_model(inputs)
        train_features.append(features)
        train_labels.append(labels)

test_features = []
test_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        # print("aa:",input)
        inputs = inputs.unsqueeze(1)
        # print("unsequzze:",input)

        features = cnn_model(inputs)
        test_features.append(features)
        test_labels.append(labels)

train_features = torch.cat(train_features).numpy()
train_labels = torch.cat(train_labels).numpy()
test_features = torch.cat(test_features).numpy()
test_labels = torch.cat(test_labels).numpy()



# PSO优化SVM参数
def svm_accuracy(params):
    C, gamma = params
    svm_model = SVC(C=C, gamma=gamma)
    svm_model.fit(train_features, train_labels)
    predictions = svm_model.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    return -accuracy  # PSO需要最小化目标函数

# PSO参数
lb = [0.1, 0.0001]  # Lower bounds for C and gamma
ub = [100, 1]       # Upper bounds for C and gamma

# 执行PSO优化
best_params, best_score = pso(svm_accuracy, lb, ub, swarmsize=30, maxiter=100)
best_C, best_gamma = best_params

# 使用最佳参数训练SVM
svm_model = SVC(C=best_C, gamma=best_gamma)
svm_model.fit(train_features, train_labels)
svm_predictions = svm_model.predict(test_features)
svm_accuracy = accuracy_score(test_labels, svm_predictions)
print("真实值：",test_labels)
print("预测值：",svm_predictions)
print(f'PSO-SVM Accuracy: {svm_accuracy}')

# 使用提取的特征训练随机森林
rf_model = RandomForestClassifier(n_estimators=500)
rf_model.fit(train_features, train_labels)
rf_predictions = rf_model.predict(test_features)
rf_accuracy = accuracy_score(test_labels, rf_predictions)
print(f'Random Forest Accuracy: {rf_accuracy}')




path =  './/Data//Cls///test_output.csv'
Nirdata = np.loadtxt(open(path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
x_newTest = Nirdata[:, :-1]
y_newTest = Nirdata[:, -1]


if(1):#    
    for i in range(x_newTest.shape[0]):
        x_newTest[i] = airpls.airPLS_deBase(x_newTest[i])
    print("对测试集已经做出了airpls")
if(1):
    min_value = np.min(x_newTest)  
    max_value = np.max(x_newTest)  
    
    # 对整个数据集进行归一化  
    normalized_data = (x_newTest - min_value) / (max_value - min_value)  
    x_newTest = normalized_data
# 将 x_newTest 转换为 PyTorch 张量
x_newTest_tensor = torch.tensor(x_newTest, dtype=torch.float32)
print(x_newTest_tensor.shape)
# 使用已经训练好的 CNN 模型提取特征
cnn_model.eval()

x_newTest_features=[]
with torch.no_grad():
    for i in range(len(x_newTest_tensor)):
        data_point = x_newTest_tensor[i]
        data_point = data_point.unsqueeze(0).unsqueeze(0)  # 添加维度以匹配网络输入
        feature = cnn_model(data_point).numpy()
        x_newTest_features.append(feature)

x_newTest_features = np.array(x_newTest_features).squeeze()
print(x_newTest_features)
svm_predictions_new = svm_model.predict(x_newTest_features)

# 打印预测结果
print("Predictions for new prediction data:", svm_predictions_new)
print("Predictions for new test data:",y_newTest)
rf_accuracy = accuracy_score(y_newTest, svm_predictions_new)
print(f'acc Accuracy: {rf_accuracy}')


rf_predictions = rf_model.predict(x_newTest_features)
rf_accuracy = accuracy_score(y_newTest, rf_predictions)
print("Predictions for new prediction data:", rf_predictions)
print("Predictions for new test data:",y_newTest)
print(f'Random Forest Accuracy: {rf_accuracy}')