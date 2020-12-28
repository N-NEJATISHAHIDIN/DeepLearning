import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np

NUM_CLASSES = 21

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
        
        self.conv2 = nn.Conv2d(3, 128, 3)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, 3)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, 3)
        self.conv4_bn = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 16, 3)
        self.conv5_bn = nn.BatchNorm2d(16)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(16 * 26 * 26, 120)
        self.fc1_bn = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.fc2_bn = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x):
        
        
        #res2 = x[:,:,1:109, 1:109]
        x = F.relu(self.conv2_bn(self.conv2(x)))
        #x += res2
        
        #res3 = self.pool(x[:,:,1:107, 1:107])
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        #x += res3
        
        #res4 = X[:,:,1:52, 1:52]
        x = self.pool(F.relu(self.conv4_bn(self.conv4(x))))
        #x += res4
        
        #res5 = self.pool( X[:,:,1:50, 1:50])
        x = self.pool(F.relu(self.conv5_bn(self.conv5(x))))
        #x += res5

        x = x.view(x.size()[0], 16 * 26 * 26)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        return x

