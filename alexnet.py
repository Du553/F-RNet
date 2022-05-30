import torch
import numpy as np
from torch.autograd import Variable

import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor
from PIL import Image

import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from pylab import *
from torchvision.datasets import ImageFolder
from pylab import *

import pickle as pkl
import pywt

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

#声明用GPU（指定具体的卡）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

my_transforms = transforms.Compose([
    transforms.Scale(40),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()
    ])

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 第一层是 5x5 的卷积，输入的channels 是 3，输出的channels是 64,步长 1,没有 padding
        # Conv2d 的第一个参数为输入通道，第二个参数为输出通道，第三个参数为卷积核大小
        # ReLU 的参数为inplace，True表示直接对输入进行修改，False表示创建新创建一个对象进行修改
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,5),
            nn.ReLU()
        )
        
        
        # 第二层为 3x3 的池化，步长为2，没有padding
        self.max_pool1 = nn.MaxPool2d(3, 2)
        
        # 第三层是 5x5 的卷积， 输入的channels 是64，输出的channels 是64，没有padding
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 5, 1),
            nn.ReLU(True)
        )
        
        #第四层是 3x3 的池化， 步长是 2，没有padding
        self.max_pool2 = nn.MaxPool2d(3,2)
        
        #第五层是全连接层，输入是 1204 ，输出是384
        self.fc1 = nn.Sequential(
            nn.Linear(1024,384),
            nn.ReLU(True)
        )
        
        # 第六层是全连接层，输入是 384， 输出是192
        self.fc2 = nn.Sequential(
            nn.Linear(384, 192),
            nn.ReLU(True)
        )
        
        # 第七层是全连接层，输入是192， 输出是 10
        self.fc3 = nn.Linear(192, 10)
        
    def forward(self, x):
        '''
        x = x.cpu().numpy()
        
        cA,(cH,cV,cD)=pywt.dwt2(x,"haar")
    #小波变换之后，低频分量对应的图像
        d1 = np.uint8(cA/np.max(cA)*255)

    # 小波变换之后，水平方向上高频分量对应的图像
        d2 = np.uint8(cH/np.max(cH)*255)
    # 小波变换之后，垂直方向上高频分量对应的图像
        d3 = np.uint8(cV/np.max(cV)*255)
    # 小波变换之后，对角线方向上高频分量对应的图像
        d4 = np.uint8(cD/np.max(cD)*255)
    #让a保持和x类型一致       
  
        d1 = torch.tensor(d1)
        d2 = torch.tensor(d2)
        d3 = torch.tensor(d3)
        d4 = torch.tensor(d4)
    
    #小波变换后拼接四张图片变成a
        a1 = torch.cat([d1,d2],dim=2)    
        a2 = torch.cat([d3,d4],dim=2)
        a = torch.cat([a1,a2],dim=3) 
        a = a.to(torch.float32)        
    
        #input输入的a也要放到gpu上
        a = a.to(device)
    '''
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)

        #将图片矩阵拉平
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x  
    
def main():

    # device = ...
    # criterion = ...
    dataset = ImageFolder('C:/Users/gisdom/Desktop/train/数据集1/train',my_transforms)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    dataset2 = ImageFolder('C:/Users/gisdom/Desktop/train/数据集1/test',my_transforms)
    test_loader = DataLoader(dataset2, batch_size=16, shuffle=True)
    
    alexnet = AlexNet()
    #将模型（model）加载到GPU上
    alexnet.to(device)
                                               
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = torch.optim.Adam(alexnet.parameters(), lr=lr)
    
    val_true = 0#临时保存准确率
    
    log_dir = os.path.join('C:/Users/gisdom/Desktop/train/参数')
    train_writer = SummaryWriter(log_dir=log_dir)
    
    for i, (images, labels) in enumerate(train_loader):
        # Training
        #losses = 0
        for epoch in range(90):  # K折交叉验证
        #把数据和标签放到GPU上
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)
    
            
        # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = alexnet(images)
            labels=labels.long()
            loss = criterion(outputs, labels)
            loss.backward()
            
            #losses = losses + loss
        
            optimizer.step()
            
        #losses = losses/10
        train_writer.add_scalar('Loss', loss, i)
        
        if (i+1)%1 == 0:
            print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" % (epoch+1,10,i+1,len(train_loader),loss.item()))
 
        #Decaying Learning Rate
        if (epoch + 1) % 3 == 0:
            lr /= 3
            optimizer = torch.optim.Adam(alexnet.parameters(), lr=lr)
        
        # Test
        correct = 0
        total = 0

        for images, labels in test_loader:
            images = Variable(images).to(device)
            outputs = alexnet(images).to(device)
    
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            
            zongshu = 0
            predicted = predicted.to(torch.int32)
            for a in range(predicted.size(0)):
                if predicted[a] == labels[a]:
                    zongshu += 1
            correct += zongshu
    


        result = correct / total
        result = 100 * result
        print('Accuracy of the model on the test images: %d %%' % result)
        
        train_writer.add_scalar('Accuracy',result, i)
        
        #保存单轮准确率最高的神经网络参数
        if val_true >= int(result):
            print("no change")
        else:
            val_true = int(result)
            print(val_true)
            # 保存神经网络的训练模型的参数
            torch.save(alexnet.state_dict(), 'chaye_alexnet.pkl') 

    #YANZHENG
    # 构建一个网络结构
    alexnet = AlexNet()
    # 将模型参数加载到新模型中
    alexnet.load_state_dict(torch.load('chaye_alexnet.pkl'))
    #将网络放到gpu上
    alexnet.to(device)                    
                      
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = torch.optim.Adam(alexnet.parameters(), lr=lr)         

    dataset3 = ImageFolder('C:/Users/gisdom/Desktop/train/数据集1/yanzheng',my_transforms)
    YZ_loader = DataLoader(dataset3, batch_size=16, shuffle=True)
            

    total = 300
    total_g = 100
  
    mex = 0#定义0矩阵
    for images, labels in YZ_loader:
        images = Variable(images).to(device)
        outputs = alexnet(images).to(device)
        
        _, predicted = torch.max(outputs.data, 1)
        #total += labels.size(0)#labels.size(0)输出labels的大小（16）
    
        C = confusion_matrix(labels.cpu().numpy(),predicted.cpu().numpy())
        mex = mex + C
        
        '''
        predicted = predicted.to(torch.int32)
        for i in range(predicted.size(0)):
            if predicted[i] == labels[i]:
                zongshu += 1
                if predicted[i]+1 == 1:
                    zongshu_1 += 1
                elif predicted[i]+1 == 2:
                    zongshu_2 += 1
                elif predicted[i]+1 == 3:
                    zongshu_3 += 1
                #elif predicted[i]+1 == 4:
                   # zongshu_4 += 1
                
        correct += zongshu
        correct_1 += zongshu_1
        correct_2 += zongshu_2
        correct_3 += zongshu_3
        #correct_4 += zongshu_4
    '''
    print(mex)#输出混淆矩阵
  
    #召回率recall
    rc1 = 100 * (mex[0,0] / total_g)
    rc2 = 100 * (mex[1,1] / total_g)
    rc3 = 100 * (mex[2,2] / total_g)
    
    
    print('recall of the 1 : %d %%' % rc1)
    print('recall of the 2 : %d %%' % rc2)
    print('recall of the 3 : %d %%' % rc3)
    
    #精确率precision
    
    pre1 = 100 * (mex[0,0] / (mex[0,0] + mex[1,0] + mex[2,0]))
    pre2 = 100 * (mex[1,1] / (mex[1,1] + mex[0,1] + mex[2,1]))
    pre3 = 100 * (mex[2,2] / (mex[2,2] + mex[0,2] + mex[1,2]))
    
    
    print('precision of the 1 : %d %%' % pre1)
    print('precision of the 2 : %d %%' % pre2)
    print('precision of the 3 : %d %%' % pre3)
    
    
    #总准确率accruacy
    acc = 100 * (((mex[0,0] + mex[1,1] + mex[2,2])) / total) 
    print('Accuracy of the model on the YANZHENG images: %d %%' % acc) 
           


if __name__ == '__main__':
    main()



