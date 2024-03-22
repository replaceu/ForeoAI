import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader

#准备数据集
class DiabetesDataset(Dataset):
    #加载数据集
    def __init__(self,filepath):
        xy = np.loadtxt(filepath,delimiter=',',dtype=np.float32,encoding='utf-8')
        #shape[0]是矩阵的行数,shape[1]是矩阵的列
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,:-1])
        self.y_data = torch.from_numpy(xy[:,[-1]])

    #获取数据索引
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    #获取数据总量
    def __len__(self):
        return self.len

dataset = DiabetesDataset('diabetes.csv')
#num_workers：多线程
train_loader = DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=2)

#定义模型
class FnnModel(torch.nn.Module):
    def __init__(self):
        super(FnnModel, self).__init__()
        # 输入数据的特征有8个,也就是有8个维度,随后将其降维到6维
        self.linear1 = torch.nn.Linear(8,6)
        #从6维降到4维
        self.linear2 = torch.nn.Linear(6,4)
        #从4维降到2维
        self.linear3 = torch.nn.Linear(4,2)
        #从2维降到1维
        self.linear4 = torch.nn.Linear(2,1)
        #可以视其为网络的一层,而不是简单的函数使用
        self.sigmod = torch.nn.Sigmoid

    def forward(self,x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        return x

model = FnnModel()

#返回损失的平均值
criterion = torch.nn.BCELoss(reduction='mean')
#返回优化器
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

epoch_list=[]
loss_list=[]

#进行训练
def training():
    for epoch in range(100):
        loss_one_epoch = 0
        # i是一个epoch中第几次迭代,一共756条数据,每个mini_batch为32,所以一个epoch需要迭代23次
        for i,data in enumerate(train_loader,0):
            inputs,labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred,labels)
            loss_one_epoch +=loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_list.append(loss_one_epoch/23)
        epoch_list.append(epoch)
        print('Epoch[{}/{}],loss:{:.6f}'.format(epoch + 1, 100, loss_one_epoch / 23))

    # Drawing
    plt.plot(epoch_list, loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()








