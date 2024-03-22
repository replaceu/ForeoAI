import torch
import torch.nn as nn

#print(torch.__version__)
# print(torch.version.cuda)
# print(torch.cuda.is_available())
#
# basic_tensor = torch.tensor([2,3])
# print(basic_tensor)
# int_tensor = torch.tensor([2,3])
# print(int_tensor)
#
# basic2_tensor = torch.Tensor(2,3)
# int2_tensor = torch.IntTensor(2,3)
# print(int2_tensor)
#
# int1_tensor = torch.randn(2,3)
# int2_tensor = torch.randn(3,2)
#
# print(int1_tensor)
# print(int2_tensor)

#mul_tensor = torch.mul(int1_tensor,int2_tensor)
# mul_tensor = torch.mm(int2_tensor,int1_tensor)
# print(mul_tensor)

tensor = torch.FloatTensor([[1,2,4,1],
                  [6,3,2,4],
                  [2,4,6,1]])


data_layer = nn.LayerNorm(4)(tensor)

print(data_layer)

