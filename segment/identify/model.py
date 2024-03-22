import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#创建FCN
class NeuralNetWork(nn.Module):
    def __init__(self,input_size,num_classes):
        super(NeuralNetWork, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size,out_features=50)
        self.fc2 = nn.Linear(in_features=50,out_features=num_classes)

    def forward(self,x):
        return self.fc2(F.relu(self.fc1(x)))

#检查形状是否正确
model = NeuralNetWork(784,10)
x = torch.rand(64,784)
print(model(x).shape)

#设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


#设置超参数
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

#加载数据
train_data = datasets.MNIST(root="dataset/",train=True,transform=transforms.ToTensor,download=True)
train_loader = DataLoader(dataset=train_data)

test_data = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

#初始化网络
model = NeuralNetWork(input_size=input_size,num_classes=num_classes).to(device=device)

#损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer  = optim.Adam(params = model.parameters(),lr=learning_rate)

#训练
for epoch in range(num_epochs):
    for batch_idx,(data,labels) in enumerate(train_loader):
        data = data.to(device = device)
        labels = labels.to(device = device)

        data = data.reshape(data.shape[0],-1)
        scores = model(data)
        loss = criterion(scores,labels)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()



