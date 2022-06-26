import torch
import torch.nn.functional as F
from torch import  nn
"""save tensor"""
x= torch.arange(4)
torch.save(x,"x-file")
x2=torch.load("x-file")
print(x2)
"""store list"""
y=torch.zeros(4)
torch.save([x,y],"xy-file")
x1,y1 = torch.load("xy-file")
print(x1,y1)
"""store dict"""
"""store paramter"""
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, x):  # 前向计算
        return self.out(F.relu(self.hidden(x)))
net = MLP()
x=torch.randn((2,20))
y=net(x)
torch.save(net.state_dict(),"mlp.params") # 文件格式为：字典  torch 默认初始化参数方法为 kaiming初始化
clone = MLP()
clone.load_state_dict(torch.load("mlp.params"))  # 实例化备份，读取参数
print(clone)
y_clone=clone(x)
print(y_clone==y)
