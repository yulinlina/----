import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))  # nn.sequential 是一种module
x = torch.rand((2, 20))
"""自定义 手动实现module"""

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, x):  # 前向计算
        return self.out(F.relu(self.hidden(x)))


# net = MLP()
# print(net(x))
"""灵活地建构网络"""

class FixedhiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.liner = nn.Linear(20, 20)

    def forward(self, x):
        x = self.liner(x)
        x = F.relu(torch.mm(x, self.rand_weight) + 1)
        x = self.liner(x)
        while x.abs().sum() > 1:
            x /= 2
        return x.sum()

"""混合搭配组合块"""

class Useless_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
        self.linear = nn.Linear(10, 20)

    def forward(self, x):
        return self.linear(self.net(x))


Mix_net = nn.Sequential(Useless_MLP(), FixedhiddenMLP())
print(Mix_net(x))
"""手动实现 sequential"""


class mysequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            self._modules[block] = block

    def forward(self, x):
        for block in self._modules.values():
            x = block(x)
        return x
