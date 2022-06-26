import numpy as np
import torch
from torch import nn
def sigmoid(z):
    return 1/(np.exp(-z))
def compute_gradient(X,Y,W):
    """
    计算梯度
    :param X: 输入
    :param Y:
    :param W:
    :return:
    """
    h =sigmoid(W.t()@X)
    dw =np.matmul((h-Y),X.t())/len(X)
    return dw
X= torch.tensor([[1,2,3],[4,5,6]])
Y =torch.tensor([0,2])
m = len(X)
n = len(X[1])
W = torch.normal(0,0.01,size = (m,n),requires_grad=True)
b = torch.zeros(n,requires_grad=True)
loss =nn.CrossEntropyLoss()
net = nn.Sequential(nn.Linear(m,n))
print(loss)

