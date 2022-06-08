import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold#10折交叉验证
import numpy as np

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

class ImageFolderSplitKFold:
    '''
    path: 保存图像的文件
    K: 划分的折数
    '''

    def __init__(self, path, K=5):
        self.path = path
        self.K = K
        self.class2num = {}
        self.num2class = {}
        self.class_nums = {}
        self.data_x_path = []
        self.data_y_label = []

        for root, dirs, files in os.walk(path):
            if len(files) == 0 and len(dirs) > 1:
                for i, dir1 in enumerate(dirs):
                    self.num2class[i] = dir1
                    self.class2num[dir1] = i
            elif len(files) > 1 and len(dirs) == 0:
                category = ""
                for key in self.class2num.keys():
                    if key in root:
                        category = key
                        break
                label = self.class2num[category]
                self.class_nums[label] = 0
                for file1 in files:
                    self.data_x_path.append(os.path.join(root, file1))
                    self.data_y_label.append(label)
                    self.class_nums[label] += 1
            else:
                raise RuntimeError("please check the folder structure!")

        self.StratifiedKFoldData = {}
        skf = StratifiedKFold(n_splits=self.K)
        skf.get_n_splits(self.data_x_path, self.data_y_label)
        print(skf)
        i = 1
        for train_index, test_index in skf.split(self.data_x_path, self.data_y_label):
            X_train, X_test = np.array(self.data_x_path)[train_index], np.array(self.data_x_path)[test_index]
            y_train, y_test = np.array(self.data_y_label)[train_index], np.array(self.data_y_label)[test_index]
            name = f'K{i}'
            self.StratifiedKFoldData[name] = ((X_train, y_train), (X_test, y_test))
            i += 1

    def getKFoldData(self):
        '''
        返回一个字典，字典里共包含K个键值对。
        keys: K1, K2, K3, ....
        values: ((x_train,y_train),(x_test,y_test))  其中的 x_train 包含K-1份，x_test包含1份
        examples:  假如K=5， 用1,2,3,4,5代表5折，
                  则：    x_train(y_train)    x_test(y_test)
                         1,2,3,4                    5
                         1,2,3,5                    4
                         1,2,4,5                    3
                         1,3,4,5                    2
                         2,3,4,5                    1
        '''
        return self.StratifiedKFoldData


class DatasetFromFilename(Dataset):
    # x: a list of image file full path
    # y: a list of image label
    def __init__(self, x, y, transforms=None):
        super(DatasetFromFilename, self).__init__()
        self.x = x
        self.y = y
        if transforms == None:
            self.transforms = ToTensor()
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = Image.open(self.x[idx])
        img = img.convert("RGB")
        return self.transforms(img), torch.tensor(self.y[idx])
    
#3*3 Convolution
def con3x3(in_channels,out_channels,stride=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=3,
                     stride=stride,padding=1,bias=False)
 
#Residual Block
class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,downsample=None):
        super(ResidualBlock,self).__init__()
        #卷积层
        self.conv1 = con3x3(in_channels,out_channels,stride)
        #平展层
        self.bn1 = nn.BatchNorm2d(out_channels)
        #激活函数
        self.relu = nn.ReLU(inplace=True)
        #inplace=True计算结果不会有影响，利用inplace计算可以节省内存，
        #同时还可以省去反复申请和释放内存的时间；但是会对原变量进行覆盖。
        self.conv2 = con3x3(out_channels,out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
 
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:#计算残差
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
 
        return out
 
#ResNet18 Model
class ResNet(nn.Module):
    def __init__(self,block,layers,num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = con3x3(3,16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block,16,layers[0])
        self.layer2 = self.make_layer(block,32,layers[0],2)
        self.layer3 = self.make_layer(block,64,layers[1],2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64,num_classes)
        
    def make_layer(self,block,out_channels,blocks,stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                con3x3(self.in_channels,out_channels,stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(block(self.in_channels,out_channels,stride,downsample))
        self.in_channels = out_channels
        for i in range(1,blocks):
            layers.append(block(out_channels,out_channels))
 
        return nn.Sequential(*layers)   

    def forward(self, x):
        #将cuda中的tensor转换成numpy，进行小波变换.
        #报错原因：numpy不能读取CUDA tensor 需要将它转化为 CPU tensor。
        #所以如果想把CUDA tensor格式的数据改成numpy时，需要先将其转换成cpu float-tensor随后再转到numpy格式
        
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
        
        #x = x.to(device)
        out = self.conv(a)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
 
        return out

def main():
   
    #数据路径
    ROOT_DATA = r'E:/茶叶识别代码/数据集/data'
    
    #加载网络模型
    resnet = ResNet(ResidualBlock, [2, 2, 2, 2])
    #将模型（model）加载到GPU上
    resnet.to(device)
                           
                      
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)
    
    ImageSplit = ImageFolderSplitKFold(ROOT_DATA, 10)
    DATA = ImageSplit.getKFoldData()
    idx_to_class = ImageSplit.num2class
    
    val_true = 0#临时保存准确率
    Max_true = 0#所有循环中最高正确率
    
    #用tensorboard实现可视化，命令:tensorboard --logdir=./
    log_dir = os.path.join('C:/Users/asus/Desktop/resnet')
    train_writer = SummaryWriter(log_dir=log_dir)
           
    Iter = 0
    for epoch in range(1):  # K折交叉验证
        for k, key in enumerate(DATA.keys()):
            
            (x_train, y_train), (x_test, y_test) = DATA[key]
            training_dataset = DatasetFromFilename(x_train, y_train, transforms=my_transforms)
            test_dataset = DatasetFromFilename(x_test, y_test, transforms=my_transforms)
            train_loader = DataLoader(training_dataset, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
            
            
        # Training
            for i, (images, labels) in enumerate(train_loader):
                #把数据和标签放到GPU上
                images = Variable(images).to(device)
                labels = Variable(labels).to(device)
    
            
            # Forward + Backward + Optimize
                optimizer.zero_grad()
                outputs = resnet(images)
                labels=labels.long()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                    
                if (i+1)%1 == 0:
                    
                   Iter = Iter+1
                   print("Epoch [%d/%d], Iter [%d] Loss: %.4f" % (epoch+1,10,Iter,loss.item()))
                   train_writer.add_scalar('Loss', loss, Iter) 
            
        #Decaying Learning Rate
            if (epoch + 1) % 3 == 0:
                lr /= 3
                optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)
            
            # Test
            correct = 0
            total = 0

            for images, labels in test_loader:
                images = Variable(images).to(device)
                outputs = resnet(images).to(device)
        
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
            
            train_writer.add_scalar('Accuracy', result, Iter)
            
            #保存单轮准确率最高的神经网络参数
            if val_true >= int(result):
                print("no change")
            else:
                val_true = int(result)
                print(val_true)
                # 保存神经网络的训练模型的参数
                torch.save(resnet.state_dict(), 'chaye_resnet.pkl') 
                
    ##YANZHENG
    # 构建一个网络结构
    resnet = ResNet(ResidualBlock, [2, 2, 2, 2])
    # 将模型参数加载到新模型中
    resnet.load_state_dict(torch.load('chaye_resnet.pkl'))
    
    #将模型（model）加载到GPU上
    resnet.to(device)
    
                      
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)         
    
    dataset3 = ImageFolder('E:/茶叶识别代码/数据集/yanzheng',my_transforms)
    YZ_loader = DataLoader(dataset3, batch_size=32, shuffle=True)
            
    
    total = 240
    total_g = 60
    
    #mex = torch.zeros(3,3)#定义3x3的0矩阵
    mex = 0
    for images, labels in YZ_loader:
        images = Variable(images)
        outputs = resnet(images)
        
        _, predicted = torch.max(outputs.data, 1)
        #total += labels.size(0)#labels.size(0)输出labels的大小（16）
        
        C = confusion_matrix(labels.cpu().numpy(),predicted.cpu().numpy())
        mex = mex + C
        
       # predicted = predicted.to(torch.int32)
       # for i in range(predicted.size(0)):
       #     if predicted[i] == labels[i]:
       #         zongshu += 1
                
       # correct += zongshu
    
    print(mex)#输出混淆矩阵
      
    #召回率recall
    rc1 = 100 * (mex[0,0] / total_g)
    rc2 = 100 * (mex[1,1] / total_g)
    rc3 = 100 * (mex[2,2] / total_g)
    rc4 = 100 * (mex[3,3] / total_g)   
    
    print('召回率recall of the 1 : %d %%' % int(rc1))
    print('召回率recall of the 2 : %d %%' % int(rc2))
    print('召回率recall of the 3 : %d %%' % int(rc3))
    print('召回率recall of the 3 : %d %%' % int(rc4))
    
    #精确率precision
    
    pre1 = 100 * (mex[0,0] / (mex[0,0] + mex[1,0] + mex[2,0] + mex[3,0]))
    pre2 = 100 * (mex[1,1] / (mex[0,1] + mex[1,1] + mex[2,1] + mex[3,1]))
    pre3 = 100 * (mex[2,2] / (mex[0,2] + mex[1,2] + mex[2,2] + mex[3,2]))
    pre4 = 100 * (mex[3,3] / (mex[0,3] + mex[1,3] + mex[2,3] + mex[3,3]))  
    
    print('精确率precision of the 1 : %d %%' % int(pre1))
    print('精确率precision of the 2 : %d %%' % int(pre2))
    print('精确率precision of the 3 : %d %%' % int(pre3))
    print('精确率precision of the 3 : %d %%' % int(pre4))
    
    #总准确率accruacy
    acc = 100 * (((mex[0,0] + mex[1,1] + mex[2,2] + mex[3,3])) / total) 
    print('Accuracy of the model on the YANZHENG images: %d %%' % int(acc)) 



if __name__ == '__main__':
    main()





