# -*- coding:utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary
from torch.autograd import Variable
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

'''
# 关于backward方法的测试 y=x^3-7x^2+11x 则 y'=3x^2-14x+11
def func(x):
	return (x**3-7*x**2+11*x)

# 定义输入
x = Variable(torch.Tensor([2.0]), requires_grad=True) # 注意需设置require_grad=True
y = func(x)
# help(y.backward)
y.backward() # retain_graph=True
print(x.grad) # -5

# 换成另一个输入变量
x = Variable(torch.Tensor([3.0]), requires_grad=True)
y = func(x)
y.backward()
print(x.grad)

# 二维变量
u = torch.tensor(2.0, requires_grad=True)
v = torch.tensor(1.0, requires_grad=True)
f = 3*u**2*v - 4*v**3
f.backward(retain_graph=True)
print(u.grad)
print(v.grad)
'''

'''
x = torch.linspace(-10.0,10.0,requires_grad=True)
x_squared = x**2
y = torch.sum(x**2)
y.backward()
plt.figure(figsize=(8,5))
plt.plot(x.detach().numpy(),x_squared.detach().numpy(),label='Function',color='blue',lw=3)
plt.plot(x.detach().numpy(),x.grad.detach().numpy(),label='Derivative',color='red',lw=3,linestyle='--')
plt.legend(fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
'''

features=torch.randn((1,3),requires_grad=True)
print(features)

n_input=features.shape[1]
n_hidden=5
n_output=1

w1=torch.randn(n_input,n_hidden)
w2=torch.randn(n_hidden,n_output)

b1=torch.randn(1,n_hidden)
b2=torch.randn(1,n_output)

def activation(x): # sigmoid
	return 1/(1+torch.exp(-x))

print("Shape of the input features: ",features.shape)
print("Shape of the first tensor of weights (between input and hidden layers): ",w1.shape)
print("Shape of the second tensor of weights (between hidden and output layers): ",w2.shape)
print("Shape of the bias tensor added to the hidden layer: ",b1.shape)
print("Shape of the bias tensor added to the output layer: ",b2.shape)

h1=activation(torch.mm(features,w1)+b1)
h2=activation(torch.mm(h1,w2)+b2)
print(h2)
print(h1)
print(h1.detach())
h2.backward()
print(features.grad)