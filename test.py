# -*-coding:utf-8-*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.autograd import Variable
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import OrderedDict

# 自定义的类 需要继承nn.Module模块
class Net(nn.Module):
	def __init__(self):
		# 子类的__init__两种初始化方法
		# nn.Module.__init__(self) # 方法一
		super(Net,self).__init__() # 方法二
		# 两个卷积层
		self.conv1 = nn.Conv2d(3, 6, 5) # in_channel out_channel kernal_size
		self.conv2 = nn.Conv2d(6, 16, 5)
		# 三个全连接层
		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def flatFeatures(self, x): # 把得到的特征值拉成一维的
		size = x.size()[1:]
		nums = 1
		for s in size:
			nums *= s
		return nums

	# 前向传播
	def forward(self, x): # x: bach_size*channel*heigth*width
		x=F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		x=F.max_pool2d(F.relu(self.conv2(x)), 2) # 此处2等价于(2,2)
		x=x.view(-1, self.flatFeatures(x)) # 第一个参数-1表示根据后边的自动推导
		x=F.relu(self.fc1(x))
		x=F.relu(self.fc2(x))
		x=self.fc3(x)
		return x

net = Net()

'''各种打印网络参数的方式'''
# print(net)
# print(net.named_parameters)
# for name, parameters in net.named_parameters():   
#     print(name, ';', parameters.size())
# print(summary(net,(3,32,32)))
# print(list(net.parameters())) #为什么是10呢？ 因为不仅有weights，还有bias， 10=5*2。
# print(len(list(net.parameters())))

input = Variable(torch.randn(3,3,32,32),requires_grad=True)
out = net(input) #这个地方就神奇了，明明没有定义__call__()函数啊，所以只能猜测是父类实现了，并且里面还调用了forward函数
print(out)       #查看源码之后，果真如此。那么，forward()是必须要声明的了，不然会报错

out.backward(torch.randn(3, 10))
# print(input.grad)

# print(net.state_dict())
#两种保存与恢复模型的方式
# 1.只保存参数
torch.save(net.state_dict(),'net_only_params.pkh')
net2=Net()
net2.load_state_dict(torch.load('net_only_params.pkh'))
# print(net2.state_dict())

# 2.保存模型参数与结构
torch.save(net,'net_with_structure.pkh')
net2=torch.load('net_with_structure.pkh')
# print(net2)

model = nn.Sequential(OrderedDict([
	('conv1',nn.Conv2d(1,20,5)),
	('relu1',nn.ReLU()),
	('conv2',nn.Conv2d(20,64,5)),
	('relu2',nn.ReLU())
]))
print(model)

# 打印网络中的参数
params=model.state_dict()
for k,v in params.items():
	print(k)    #打印网络中的变量名
for p in model.parameters():
	print(p)
# print(params['conv1.weight'])   #打印conv1的weight
# print(params['conv1.bias'])   #打印conv1的bias