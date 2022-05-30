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

cfg = {
  'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
  'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
  'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
  'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# 模型需继承nn.Module
class VGG(nn.Module):
# 初始化参数：
  def __init__(self, vgg_name):
    super(VGG, self).__init__()
    self.features = self._make_layers(cfg[vgg_name])
    self.classifier = nn.Linear(512, 10)
 
# 模型计算时的前向过程，也就是按照这个过程进行计算
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
    out = self.features(x)
    out = out.view(out.size(0), -1)
    out = self.classifier(out)
    return out
 
  def _make_layers(self, cfg):
    layers = []
    in_channels = 3
    for x in cfg:
      if x == 'M':
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
      else:
        layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
              nn.BatchNorm2d(x),
              nn.ReLU(inplace=True)]
        in_channels = x
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    return nn.Sequential(*layers)

def main():
    
    # device = ...
    # criterion = ...
    dataset = ImageFolder('C:/Users/gisdom/Desktop/train/数据集1/train',my_transforms)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    dataset2 = ImageFolder('C:/Users/gisdom/Desktop/train/数据集1/test',my_transforms)
    test_loader = DataLoader(dataset2, batch_size=16, shuffle=True)
    
    vggnet = VGG("VGG16")
    #将模型（model）加载到GPU上
    vggnet.to(device)
                                               
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = torch.optim.Adam(vggnet.parameters(), lr=lr)
    
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
            outputs = vggnet(images)
            labels=labels.long()
            loss = criterion(outputs, labels)
            loss.backward()
            
            #losses = losses + loss
            
            optimizer.step()
        
       # losses = losses/10
        train_writer.add_scalar('Loss', loss, i)

        if (i+1)%1 == 0:
           print("Epoch %d, Loss: %.4f" % (i+1,loss.item()))
     
        #Decaying Learning Rate
        if (i + 1) % 3 == 0:
            lr /= 3
        optimizer = torch.optim.Adam(vggnet.parameters(), lr=lr)
        
        # Test
        correct = 0
        total = 0

        for images, labels in test_loader:
            images = Variable(images).to(device)
            outputs = vggnet(images).to(device)
    
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
            torch.save(vggnet.state_dict(), 'chaye_vggnet.pkl') 

    #YANZHENG
    # 构建一个网络结构
    vggnet = VGG("VGG16")
    # 将模型参数加载到新模型中e
    vggnet.load_state_dict(torch.load('chaye_vggnet.pkl'))
    #将网络放到gpu上
    vggnet.to(device)                    
                      
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = torch.optim.Adam(vggnet.parameters(), lr=lr)         

    dataset3 = ImageFolder('C:/Users/gisdom/Desktop/train/数据集1/yanzheng',my_transforms)
    YZ_loader = DataLoader(dataset3, batch_size=16, shuffle=True)
            
    total = 300
    total_g = 100
    mex = 0#定义0矩阵
    for images, labels in YZ_loader:
        images = Variable(images).to(device)
        outputs = vggnet(images).to(device)
        
        _, predicted = torch.max(outputs.data, 1)
       # total += labels.size(0)#labels.size(0)输出labels的大小（16）
        
        C = confusion_matrix(labels.cpu().numpy(),predicted.cpu().numpy())
        mex = mex + C
        
        '''
        predicted = predicted.to(torch.int32)
        for i in range(predicted.size(0)):
            if predicted[i] == labels[i]:
                zongshu += 1
                            
        correct += zongshu
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



