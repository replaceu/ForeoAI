import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from PIL import Image

# #存放所有样本标签
# all_skus = []
#
# #读取所有类别名称
# category = pd.read_table('dataset/category.txt')
# #列表中列的顺序
# column_order = ['type','img','label','x1','y1','x2','y2']
# #遍历目录1~100,读取所有图片的标签信息，汇集到all_skus列表
# for i in range(1,101,1):
#     #读取当前目录i的标签信息
#     skus = pd.read_table(f'dataset/{i}/bb_info.txt', header=0, sep='\s+')
#     #将图像id映射为对应的文件路径
#     skus['img'] = skus['img'].apply(lambda x:f'./dataset/{i}/'+str(x)+'.jpg')
#     #新增一列label，标注图片类别名称
#     skus['label'] = skus.apply(lambda x:category['name'][i-1],axis=1)
#     skus['type'] = skus.apply(lambda x:'',axis=1)
#     skus = skus[column_order]
#     #保存当前类别的标签文件
#     skus.to_csv(f'./dataset/{i}/label.csv',index=None,header=['type','img','label','x1','y1','x2','y2'])
#     #汇聚到all_skus
#     all_skus.extend(np.array(skus).tolist())
#
# #保存列表到文件中
# df_skus = pd.DataFrame(all_skus)
# df_skus.to_csv('./dataset/datasets.csv',index=None,header=['type','img','label','x1','y1','x2','y2'])

#随机洗牌，打乱数据排列顺序，划分为train，validate，test
datasets = pd.read_csv('dataset/dataset.csv')
datasets = shuffle(datasets,random_state=2022)
datasets = pd.DataFrame(datasets).reset_index(drop=True)

#总行数
rows = datasets.shape[0]
#测试样本集数量
test_n = rows//40
#验证样本数量
validate_n=rows//5
train_n = rows-test_n-validate_n
print(f'测试样本数:{test_n},验证集样本数:{validate_n},训练集样本数:{train_n}')
#按照一定比例对数据集划分
for row in range(test_n):
    datasets.iloc[row,0] = 'TEST'
for row in range(validate_n):
    datasets.iloc[row+test_n,0]='VALIDATE'
for row in range(train_n):
    datasets.iloc[row+validate_n+test_n,0] = 'TRAIN'

#将Bounding Box的坐标改为浮点类型，取值范围[0,1]
print('开始对BBox的坐标进行归一化调整，请耐心等待...')

for row in range(rows):
    #print(datasets)
    #读取图像
    #print(datasets.iloc[row,1])
    img = Image.open(datasets.iloc[row,1])
    (width,height) = img.size
    width = float(width)
    height = float(height)
    datasets.iloc[row,3] =round(datasets.iloc[row,3]/width,3)
    datasets.iloc[row,4] = round(datasets.iloc[row,4] /height, 3)
    datasets.iloc[row,5] = round(datasets.iloc[row,5] / width, 3)
    datasets.iloc[row,6] = round(datasets.iloc[row,6] /height, 3)
#插入空列
datasets.insert(datasets.shape[1],'Null1','')
datasets.insert(datasets.shape[1],'Null2','')
#调整列的顺序，为以后数据集划分做准备
order = ['type','img','label','x1','y1','Null1','Null2','x2','y2']
datasets = datasets[order]
print(datasets.head())
datasets.to_csv('./dataset/datasets.csv',index=None,header=None)
print('数据集构建完毕!')


