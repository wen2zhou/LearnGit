"""
---------------------------------------------------------------------
Training an image classifier
---------------------------------------------------------------------
For this assingment you'll do the following steps in order:
1. Load and normalizing the CIFAR10 training and test datasets using
   ``torchvision``
2. Define a Convolutional Neural Network (at least 4 conv layer)
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data
---------------------------------------------------------------------
"""

# IMPORTING REQUIRED PACKAGES
import os
import numpy as np
import scipy.io as sio
import torch
import torchvision
import torchvision.transforms as transforms

# DEFINE VARIABLE
BATCH_SIZE = 32                 # YOU MAY CHANGE THIS VALUE  32
EPOCH_NUM = 80                  # YOU MAY CHANGE THIS VALUE  80
LR = 0.002                      # YOU MAY CHANGE THIS VALUE  0.002
MODEL_SAVE_PATH = './Models'

if not os.path.exists(MODEL_SAVE_PATH):
    os.mkdir(MODEL_SAVE_PATH)

# DEFINING TRANSFORM TO APPLY TO THE IMAGES
# YOU MAY ADD OTHER TRANSFORMS FOR DATA AUGMENTATION
    
transform_dt = transforms.Compose(
    [transforms.Resize(32),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


########################################################################
# 1. LOAD AND NORMALIZE CIFAR10 DATASET
########################################################################

#FILL IN: Get train and test dataset and create respective dataloader
trainset = torchvision.datasets.CIFAR10(
        root = MODEL_SAVE_PATH,
        train = True,
        download = True,
        transform = transform_dt
        )

trainloader = torch.utils.data.DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)

testset = torchvision.datasets.CIFAR10(
        root = MODEL_SAVE_PATH,
        train = False,
        download = True,
        transform = transform_dt
        )

testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

########################################################################
# 2. DEFINE YOUR CONVOLUTIONAL NEURAL NETWORK AND IMPORT IT
########################################################################

'''
import sys
cnn_model_path = '../cnn_model.py'
sys.path.append(os.path.abspath(cnn_model_path))

import cnn_model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = cnn_model.ConvNet().to(device)
'''

########################################################################
# 2. DEFINE YOUR CONVOLUTIONAL NEURAL NETWORK
########################################################################

import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, init_weights=False):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)  # (32-5)/1 + 1 = 28
        self.conv2 = nn.Conv2d(64, 64, 5) # (28-5)/1 + 1 = 24
        self.pool1 = nn.MaxPool2d(2)      # 24/2=12
        self.conv3 = nn.Conv2d(64,64,3) # (12-3)/1 + 1 = 10  (64,128,3)
        self.conv4 = nn.Conv2d(64,64,3) # (10-3)/1 + 1 = 8  (128,128,3)
        self.pool2 = nn.MaxPool2d(2)      # 8/2=4
        self.drop_out = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=64*4*4, out_features=512, bias=False)
        self.fc2 = nn.Linear(in_features=512, out_features=10, bias=False)
        # final out_features=10
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.conv1(x)) 
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv3(x))      
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
      
        x = x.reshape(-1, 64*4*4)  # ！！！！！！！！！！！！！！！
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        out = F.softmax(x, dim=1)
        
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = ConvNet().to(device)
print(device,"\n",net)


########################################################################
# 3. DEFINE A LOSS FUNCTION AND OPTIMIZER
########################################################################

import torch.optim as optim

#FILL IN : the criteria for ce loss
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)


#########################################################################
## 4. TRAUB THE NETWORK
#########################################################################

test_accuracy = []
train_accuracy = []
train_loss = []

for epoch in range(EPOCH_NUM):  # loop over the dataset multiple times

   running_loss = 0.0
   test_min_acc = 0
   total = 0
   correct = 0

   for i, data in enumerate(trainloader, 0):
       # get the inputs
       inputs, labels = data

       # zero the parameter gradients
       optimizer.zero_grad()

       # forward + backward + optimize
       outputs = net(inputs.to(device))
       loss = criterion(outputs, labels.to(device))
       loss.backward()
       optimizer.step()

       # print statistics
       running_loss += loss.item()
       _, predicted = torch.max(outputs.data, 1)

       # FILL IN: Obtain accuracy for the given batch of data using
       # the formula acc = 100.0 * correct / total where 
       # total is the toal number of images processed so far
       # correct is the correctly classified images so far
       
       total += inputs.shape[0]

       pred_labels = predicted
       correct += pred_labels.eq(labels.to(device)).sum().item()

       train_loss.append(running_loss/20)
       train_accuracy.append(100.0*correct/total)

       if i % 20 == 19:  # print every 20 mini-batches
           print('Train: [%d, %5d] loss: %.3f acc: %.3f' %
                 (epoch + 1, i + 1, running_loss / 20,100.0*correct/total))
           running_loss = 0.0
          
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
         #YOUR CODE HERE
          images, labels = data
          images, labels = images.to(device), labels.to(device)
          outputs = net(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
          test_accuracy.append(100.0*correct/total)

print(test_accuracy)
print(test_accuracy[-1])
print(max(test_accuracy))
print(min(test_accuracy))