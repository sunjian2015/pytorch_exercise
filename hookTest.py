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
