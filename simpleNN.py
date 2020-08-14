# -*-coding:utf-8-*-

import torch
import numpy as np
from sklearn.datasets import make_classification, make_regression
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchsummary import summary
from torch.autograd import Variable
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from itertools import combinations

print(torch.cuda.is_available())

input_dim = 5
data_points = 100
# 生成训练数据集
X, y = make_classification(data_points, input_dim, n_informative=3, random_state=101)
print(X.shape)
print(y.shape)
print(y)
X = X.astype(np.float32)
y = y.astype(np.float32)
comb_list = list(combinations([v for v in range(5)], 2))
print(comb_list)

# fig,ax=plt.subplots(5,2,figsize=(10,18))
# axes=ax.ravel()
# for i,c in enumerate(comb_list):
#     j,k=c
#     axes[i].scatter(X[:,j],X[:,k],c=y,edgecolor='k',s=200)
#     axes[i].set_xlabel('X'+str(j),fontsize=15)
#     axes[i].set_ylabel('X'+str(k),fontsize=15)
# plt.show()

X = torch.from_numpy(X)
# X.requires_grad=True
y = torch.from_numpy(y)
n_input = X.shape[1]  # 输入特征的维度
n_hidden1 = 8  # 第一个隐层
n_hidden2 = 4  # 第二个隐层
n_output = 1  # 输出层


class NetWork(nn.Module):
    def __init__(self):
        super(NetWork, self).__init__()
        self.hidden1 = nn.Linear(n_input, n_hidden1)
        self.hidden2 = nn.Linear(n_hidden1, n_hidden2)
        self.relu = nn.ReLU()
        self.output = nn.Linear(n_hidden2, n_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, **kwargs):  # forward为必须定义的函数
        X = self.hidden1(X)
        X = self.relu(X)
        X = self.hidden2(X)
        X = self.relu(X)
        X = self.output(X)
        X = self.sigmoid(X)
        return X


model = NetWork()
print(model)

criterion = nn.BCELoss()  # 定义损失函数，此处为二元交叉熵损失

logits = model.forward(X)  # Output of the forward pass (logits i.e. probabilities)
# print("First 10 probabilities...\n",logits[:10])
# logits_numpy = model.forward(X).detach().numpy().flatten()
# plt.figure(figsize=(15,3))
# plt.title("Output probabilities with the untrained model",fontsize=16)
# plt.bar([i for i in range(100)],height=logits_numpy)
# plt.show()

# 计算损失
loss = criterion(logits.squeeze(-1), y)
print('loss:', loss.item())
print(logits.view(-1).shape,y.shape)

print(
"Gradients of the weights of the 2nd hidden layer connections before computing gradient:\n", model.hidden2.weight.grad)
loss.backward()  # Compute gradients
print(
"Gradients of the weights of the 2nd hidden layer connections after computing gradient:\n", model.hidden2.weight.grad)

# 设置优化器
optimizer = optim.SGD(model.parameters(), lr=0.1)

optimizer.zero_grad()  # 把梯度清0，不累加
output = model.forward(X) # 一次前向传播
init_weights=model.hidden2.weight
print("Initial weights:\n",init_weights)

loss=criterion(output.squeeze(-1),y) # 计算损失
loss.backward() # 一次反向传播
grads=model.hidden2.weight.grad
print("Gradient:\n",grads)

optimizer.step() # 一次优化
updated_weights = model.hidden2.weight # 更新之后的权重
print("Updated weights:\n",updated_weights)

# 多次迭代
epochs=100
for i,e in enumerate(range(epochs)):
    optimizer.zero_grad() # 梯度清零
    output=model.forward(X) # 前向传播
    # print(X.grad) # 若X的require_grad未设置为True则此处为None
    loss=criterion(output.view(-1),y)
    print("Epoch - {}, Loss - {}".format(i+1,round(loss.item(),3)))
    loss.backward()
    optimizer.step()

'''
# 绘制loss曲线
epochs=1000
running_loss=[]
for i,e in enumerate(range(epochs)):
    optimizer.zero_grad()
    output=model.forward(X)
    loss=criterion(output.view(-1),y)
    loss.backward()
    optimizer.step()
    running_loss.append(loss.item())
plt.figure(figsize=(7,4))
plt.title("Loss over epochs",fontsize=16)
plt.plot([e for e in range(epochs)],running_loss)
plt.grid(True)
plt.xlabel("Epochs",fontsize=15)
plt.ylabel("Training loss",fontsize=15)
plt.show()
'''

