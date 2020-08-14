# -*- coding:utf-8 -*-

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision

# w1 = Variable(torch.Tensor([1.0,2.0,3.0]),requires_grad=True)#需要求导的话，requires_grad=True属性是必须的。
# w2 = Variable(torch.Tensor([1.0,2.0,3.0]),requires_grad=True)

w1 = torch.tensor([1.0,2.0,3.0],requires_grad=True)#需要求导的话，requires_grad=True属性是必须的。
w2 = torch.tensor([1.0,2.0,3.0],requires_grad=True)

print(w1.grad)
print(w2.grad)

d=torch.mean(w1)
d.backward() # backward()操作会自动累加梯度
print(w1.grad)
# w1.grad.zero_() #防止梯度累加
d.backward()
print(w1.grad)

# 利用梯度更新参数
learning_rate = 0.1
#w1.data -= learning_rate * w1.grad.data 与下面式子等价
w1.grad.zero_()
d.backward()
w1.data.sub_(learning_rate*w1.grad.data)# w1.data是获取保存weights的Tensor
print(w1)

# import torch.cuda as cuda
# 只使用部分 Variable 求出来的 loss对于原Variable求导得到的梯度
print(torch.ones(2,3))
w=Variable(torch.ones(2,3),requires_grad=True)
res = torch.mean(w[1])# 只用了variable的第二行参数
res.backward()
print(w.grad)

# 部分求导
x = Variable(torch.randn(5, 5))
y = Variable(torch.randn(5, 5))
z = Variable(torch.randn(5, 5), requires_grad=True)
a = x + y # x, y的 requires_grad的标记都为false， 所以输出的变量requires_grad也为false
print(a.requires_grad)
b = y + z
print(b.requires_grad)

# 冻结部分网络
model = torchvision.models.resnet18(pretrained=True)
print(model)
for param in model.parameters():
    param.requies_grad = False
model.fc = nn.Linear(512, 100)
optimize = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)

# volatile - 已经被弃用
j = Variable(torch.randn(5,5),volatile=True)
k = Variable(torch.randn(5,5))
m = Variable(torch.randn(5,5))
n = k+m # k,m变量的volatile标记都为False，输出的Variable的volatile标记也为false
print(n.volatile)
