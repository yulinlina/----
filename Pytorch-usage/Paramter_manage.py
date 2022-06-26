import torch
from torch import nn
from torch.nn import functional as F
net = nn.Sequential(nn.Linear(4,8),
                    nn.ReLU(),
                    nn.Linear(8,1))
x= torch.rand((2,4)) # 可以认为是list
# print(net(x))
"""visit the paramter"""
#OrderedDict([('weight', tensor([[ 0.0131, -0.0262, -0.1767,  0.3098,  0.1620, -0.0045,  0.3201,  0.2254]])),
# ('bias', tensor([-0.2713]))])
# print(net[2].state_dict())
"""visit the specific para"""
# print(type(net[2].bias))  # <class 'torch.nn.parameter.Parameter'> Parameter containing:
# print(net[2].bias)  # tensor([0.1272], requires_grad=True)
# print(net[2].bias.data)  # tensor([-0.2385])
# print(net[2].weight.grad==None) # True 还没有计算
# print(net.state_dict()['2.bias'].data) # tensor([0.1617])
"""visit all the para on one time"""
# print("visit all the para on one time")
# print(*[(name,param_shape) for name,param_shape in net.named_parameters()])
"""嵌套块访问参数"""
def block1():
    return  nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,4),nn.ReLU())
def block2():
    net =nn.Sequential()
    for i in range(4):
        net.add_module(f"block{i}",block1())
    return net
rgnet = nn.Sequential(block2(),nn.Linear(4,1))
# (rgnet(x))  # tensor([[-0.0428], [-0.0428]], grad_fn=<AddmmBackward>)
# print(rgnet)
"""内置初始化"""
def init_normal(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,mean=0,std=0.01)  # nn.init.constant(m.weight,1)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
# print(net[0].weight.data[0])
# net[0].weight.data[:]+=1
# net[0].weight.data[0,0]=42
# print(net[0].weight.data)
"""自定义初始化"""
"""参数绑定"""
