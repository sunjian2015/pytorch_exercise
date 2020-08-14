# -*- coding:utf-8 -*-

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision

# k.backward(parameters)接受的参数parameters必须要和k的大小一模一样，然后作为k的系数传回去
x = Variable(torch.ones(2, 2), requires_grad=True)
y = x + 1
# y.backward()  # 这样会报错
y.backward(torch.ones(2,2))
print(x.grad)

m = Variable(torch.FloatTensor([2, 3]), requires_grad=True)
n = Variable(torch.zeros(2))
n[0] = m[0] ** 2
n[1] = m[1] ** 3
n.backward(torch.FloatTensor([1,1]))
print(m.grad)

# 当输出有两个变量时
# y=[y1,y2]=[x1^2+3*x2, x2^2+2*x1]
m = Variable(torch.FloatTensor([2, 3]), requires_grad=True)
k = Variable(torch.zeros(2))
# m.grad.zero_()
k[0] = m[0] ** 2 + 3 * m[1]
k[1] = m[1] ** 2 + 2 * m[0]
k.backward(torch.FloatTensor([1,1]),retain_graph=True)
print(m.grad) #此处得到的结果是y的分量(y1和y2)对x1的偏导数之和，x2也是如此
# 若想得到y1对x1和x2的偏导数与y2对x1和x2的偏导数，需要如下计算方式
m.grad.zero_()
j=torch.zeros(2,2)
k.backward(torch.FloatTensor([1,0]),retain_graph=True) # y1对x1，x2求偏导数
j[:,0]=m.grad.data
m.grad.zero_()
k.backward(torch.FloatTensor([0,1])) # y2对x1，x2求偏导数
j[:,1]=m.grad.data
print(j)

# 例2
x = torch.FloatTensor([2, 1]).view(1, 2)
x = Variable(x, requires_grad=True)
y = Variable(torch.FloatTensor([[1, 2], [3, 4]]))
z = torch.mm(x, y)
jacobian = torch.zeros((2, 2))
z.backward(torch.FloatTensor([[1, 0]]), retain_graph=True)  # dz1/dx1, dz1/dx2
jacobian[:, 0] = x.grad.data
x.grad.data.zero_()
z.backward(torch.FloatTensor([[0, 1]]))  # dz2/dx1, dz2/dx2
jacobian[:, 1] = x.grad.data
print('=========jacobian========')
print('x')
print(x.data)
print('y')
print(y.data)
print('compute result')
print(z.data)
print('jacobian matrix is')
print(jacobian)