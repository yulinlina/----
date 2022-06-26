import  torch
x=torch.arange(24).reshape(2,3,4)
print(x)
sum = x.sum(axis=0)    # 按照某一个维度求和
print(sum)
y=x.clone()
# hadamard product
# print(x*y)
x=torch.tensor([[1,2,3,4],[1,3,5,7]])
print(x.sum(axis=1,keepdims=True)) # 保留该维度
# 如何求范数   几何平均
z=torch.tensor([2,4,6,8],dtype=float )
print(torch.norm(z))