import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # 1st block
        self.conv1 = nn.Conv2d(1, 32, 5)
#         self.relu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.drop1 = nn.Dropout(p=0.1)
        self.bnrm1 = nn.BatchNorm2d(32)
        
        # 2nd block
        self.conv2 = nn.Conv2d(32, 64, 5)
#         self.relu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.drop2 = nn.Dropout(p=0.1)
        self.bnrm2 = nn.BatchNorm2d(64)
        
        # 3rd block
        self.conv3 = nn.Conv2d(64, 128, 5)
#         self.relu3 = nn.PReLU()
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.drop3 = nn.Dropout(p=0.1)
        self.bnrm3 = nn.BatchNorm2d(128)
        
        # 4th block
        self.conv4 = nn.Conv2d(128, 256, 5)
#         self.relu4 = nn.PReLU()
        self.pool4 = nn.MaxPool2d(2, stride=2)
        self.drop4 = nn.Dropout(p=0.1)
        self.bnrm4 = nn.BatchNorm2d(256)
        
        # 5th block
        self.dense1 = nn.Linear(25600, 1024)
#         self.relu5 = nn.PReLU()
        self.drop5 = nn.Dropout(p=0.1)
        self.bnrm5 = nn.BatchNorm1d(1024)
        
        # 6th block
        self.dense2 = nn.Linear(1024, 1024)
#         self.relu6 = nn.PReLU()
        self.drop6 = nn.Dropout(p=0.1)
        self.bnrm6 = nn.BatchNorm1d(1024)
        
        self.out = nn.Linear(1024, 136)
        
        
    def forward(self, x):
        
        x1 = self.bnrm1(self.drop1(self.pool1(F.relu(self.conv1(x)))))
        x2 = self.bnrm2(self.drop2(self.pool2(F.relu(self.conv2(x1)))))
        x3 = self.bnrm3(self.drop3(self.pool3(F.relu(self.conv3(x2)))))
        x4 = self.bnrm4(self.drop4(self.pool4(F.relu(self.conv4(x3)))))
        x5 = self.bnrm5(self.drop5(F.relu(self.dense1(x4.reshape(x4.size()[0], -1)))))
        x6 = self.bnrm6(self.drop6(F.relu(self.dense2(x5))))
        
        x_out = self.out(x6)
        
        return x_out

    
class resnet18(nn.Module):
    
    def __init__(self):
        super(resnet18, self).__init__()
        
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.conv1 = nn.Conv2d(1, 64, 7, bias=False)
        self.resnet18.fc = nn.Linear(32768, 136)
                        
    def forward(self, x):
        
        x = self.resnet18(x)
        
        return x
