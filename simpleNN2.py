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

'''
分类测试
'''
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

model=NetWork()
optimizer=optim.SGD(model.parameters(),lr=0.1)
criterion=nn.BCELoss()
epochs=200
running_loss=[]

# logits=model.forward(X).detach().numpy().flatten()
# plt.figure(figsize=(15,3))
# plt.title("Output probabilities before training")
# plt.bar([i for i in range(100)],height=logits)
# plt.show()

# for i,e in enumerate(range(epochs)):
#     optimizer.zero_grad()
#     output = model.forward(X)
#     loss = criterion(output.view(-1),y)
#     loss.backward()
#     optimizer.step()
#     running_loss.append(loss.item())
#     if i!=0 and (i+1)%20==0:
#         logits = model.forward(X).detach().numpy().flatten()
#         plt.figure(figsize=(15,3))
#         plt.title("Output probabilities after {} epochs".format(i+1))
#         plt.bar([i for i in range(100)],height=logits)
#         plt.show()

def trainNN(model,X,y,opt,epochs=10,verbose=True):
    for i, e in enumerate(range(epochs)):
        opt.zero_grad()
        output=model.forward(X)
        loss=criterion(output.view(output.shape[0]),y)
        if verbose:
            print("Epoch - {}, Loss - {}".format(i+1,round(loss.item(), 3)))
        loss.backward()
        opt.step()
    return model

new_model= NetWork()
optimizer = optim.SGD(new_model.parameters(),lr=0.04)
trained_model=trainNN(new_model,X,y,optimizer,epochs=200,verbose=False)

class FunkyNetwork(nn.Module):
    def __init__(self):
        super(FunkyNetwork,self).__init__()
        self.hidden1=nn.Linear(n_input,n_hidden1)
        self.hidden2=nn.Linear(n_hidden1,n_hidden2)
        self.relu=nn.ReLU()
        self.tanh=nn.Tanh()
        self.output=nn.Linear(n_hidden2,n_output)
        self.sigmoid=nn.Sigmoid()
    def forward(self,X):
        X=self.hidden1(X)
        X1=self.relu(X)
        X2=self.tanh(X)
        X3=X1+X2
        X3=self.hidden2(X3)
        X3=self.relu(X3)
        X3=self.output(X3)
        X3=self.sigmoid(X3)
        return X3

new_model=FunkyNetwork()
optimizer=optim.SGD(new_model.parameters(),lr=0.04)
trained_model=trainNN(new_model,X,y,optimizer,epochs=200,verbose=False)
loss=criterion(trained_model(X).view(-1),y)
print(loss.item())

new_model= NetWork()
optimizer = optim.SGD(new_model.parameters(),lr=0.06)
trained_model=trainNN(new_model,X,y,optimizer,epochs=200,verbose=False)
loss = criterion(trained_model(X).view(trained_model(X).shape[0]),y)
print(loss.item())

'''
回归测试
'''

X,y=make_regression(200,20,20,1,noise=0.01,shuffle=False)
X = X.astype(np.float32)
y = y.astype(np.float32)
X = torch.from_numpy(X)
y = torch.from_numpy(y)
print(X.shape,y.shape)

# 标准化
X = X/X.max()
y = y/y.max()

n_input=X.shape[1]
n_hidden1=40
n_hidden2=40
n_output=1

class RegNetwork(nn.Module):
    def __init__(self):
        super(RegNetwork, self).__init__()
        self.hidden1 = nn.Linear(n_input,n_hidden1)
        self.hidden2 = nn.Linear(n_hidden1,n_hidden2)
        self.relu=nn.ReLU()
        self.output=nn.Linear(n_hidden2,n_output)

    def forward(self, X):
        X=self.hidden1(X)
        X=self.relu(X)
        X=self.hidden2(X)
        X=self.relu(X)
        X=self.output(X)
        return X

reg_model=RegNetwork()
#比较两种不同loss
def mean_quartic_error(output, target):
    """
    Computes 4-th power loss
    """
    loss = torch.mean((output - target) ** 4)
    return loss

print('***********************')
mse_loss=nn.MSELoss()
epochs=2000
running_loss = []
reg_model= RegNetwork()
optimizer = optim.SGD(reg_model.parameters(),lr=0.01)
first=True
for i,e in enumerate(range(epochs)):
    optimizer.zero_grad()
    output = reg_model.forward(X)
    if first:
        print(y.shape)
        print(y.unsqueeze(1).shape)
        print(output.shape)
        print(output.requires_grad)
        print(output.view(-1).requires_grad)
        first=False
    loss = mse_loss(output,y)
    loss.backward()
    optimizer.step()
    running_loss.append(loss.item())

plt.figure(figsize=(7,4))
plt.title("Loss over epochs with MSE loss function",fontsize=18)
plt.plot(range(epochs),running_loss)
plt.grid(True)
plt.xlabel("Epochs",fontsize=15)
plt.ylabel("Training loss",fontsize=15)
plt.show()

# epochs = 200
# running_loss = []
# reg_model= RegNetwork()
# optimizer = optim.SGD(reg_model.parameters(),lr=0.01)
# for i,e in enumerate(range(epochs)):
#     optimizer.zero_grad()
#     output = reg_model.forward(X)
#     loss = mean_quartic_error(output,y)
#     loss.backward()
#     optimizer.step()
#     running_loss.append(loss.item())
#
# plt.figure(figsize=(7,4))
# plt.title("Loss over epochs with 4-th degree loss function",fontsize=18)
# plt.plot([e for e in range(epochs)],running_loss)
# plt.grid(True)
# plt.xlabel("Epochs",fontsize=15)
# plt.ylabel("Training loss",fontsize=15)
# plt.show()