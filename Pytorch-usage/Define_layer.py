import torch
import torch.nn.functional as F
from torch import  nn
class Center_Layer(nn.Module): # 无参数
    def __init__(self):
        super(Center_Layer, self).__init__()
    def forward(self,x):
        return x-x.mean()
x = torch.FloatTensor([1,2,3,4,5])
layer = Center_Layer()
print(layer(x))

"""将自定义层放在更复杂的模型"""
net = nn.Sequential(nn.Linear(8,16),Center_Layer()) # 这里没有实例化
y = torch.randn(4,8)
print(net(y).mean())


class MyLinear(nn.Module):
    def __init__(self,in_units,units):
        super(MyLinear, self).__init__()
        self.weight= nn.Parameter(torch.randn(in_units,units))
        self.bias = nn.Parameter(torch.randn(units,))# 写法疑问
    def forward(self,x):
        linear = torch.matmul(x,self.weight.data)+self.bias.data  # 这里一定是要.data 否则其是一个类
        return F.relu(linear)
dense =MyLinear(5,3)  # 输入5个神经元 ，输出3个
print(dense.weight)
