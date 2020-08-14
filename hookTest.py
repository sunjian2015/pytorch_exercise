# -*- coding:utf-8 -*-

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision

'''
注册hook相关函数有3个：
1.  torch.autograd.Variable.register_hook
2.  torch.nn.Module.register_backward_hook
3.  torch.nn.Module.register_forward_hook
第一个是register_hook：是针对Variable对象的
后两个是：register_backward_hook和register_forward_hook是针对nn.Module这个对象的。
'''

# x=[x1, x2]
# y=x+2=[x1+2, x2+2]
# z=1/2*(y1^2+y2^2)
x = Variable(torch.randn(2, 1), requires_grad=True)
y = x+2
z = torch.mean(torch.pow(y, 2))
lr = 1e-3
z.backward(retain_graph=True)
print(x.grad)
print(y.grad) # 此处得到的是None，因为当初开发时设计的是，对于中间变量，一旦它们完成了自身反传的使命，就会被释放掉。
# x.data -= lr*x.grad.data

# 此时hook便起到作用了，简而言之，register_hook的作用是，当反传时，除了完成原有的反传，额外多完成一些任务。
# 你可以定义一个中间变量的hook，将它的grad值打印出来，当然你也可以定义一个全局列表，将每次的grad值添加到里面去。
# 需要注意的是，register_hook函数接收的是一个函数，这个函数有如下的形式：
# hook(grad) -> Variable or None
# 也就是说，这个函数是拥有改变梯度值的威力的！
grad_list = []
def print_grad(grad):
    grad_list.append(grad)
x.grad.zero_()
y.register_hook(print_grad)
z.backward()
print(grad_list)
