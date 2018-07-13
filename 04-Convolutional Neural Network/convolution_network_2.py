# -*- coding: utf-8 -*-
"""
Created on Wed May 30 19:51:21 2018

@author: hubinbin
"""

import torch
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import time

batch_size = 128
learning_rate = 1e-3
num_epoches = 20

train_dataset = datasets.MNIST(
    root='../data',train=True,transform=transforms.ToTensor(),download=False)
test_dataset = datasets.MNIST(
    root='../data',train=False,transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

class Cnn(nn.Module):
    """docstring for Cnn"""
    def __init__(self, in_dim,n_class):
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6, 16,5,stride=1,padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
        )

        self.fc=nn.Sequential(
            nn.Linear(400,120),
            nn.Linear(120,84),
            nn.Linear(84,n_class)
            )

    def forward(self,x):
        out = self.conv(x)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out

model = Cnn(1,10) # 28*28
use_gpu = torch.cuda.is_available()
if use_gpu:
    model=model.cuda()

criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=learning_rate)

for epoch in range(num_epoches):
    print('*'*10)
    print('epoch {}'.format(epoch+1))
    since=time.time()
    running_loss=0.0
    running_acc=0.0

    for i,data in enumerate(train_loader,1):
        img,lable=data
        #img=img.view(img.size(0),-1)
        if use_gpu:
            img=img.cuda()
            lable=lable.cuda()
        out = model(img)
        loss=criterion(out,lable)
        running_loss+=loss.item()*lable.size(0)
        _,pred=torch.max(out,1) # 1:index
        num_correct=(pred==lable).sum()
        running_acc+=num_correct.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%300 == 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, num_epoches, running_loss / (batch_size * i),
                running_acc / (batch_size * i)))
    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(
            train_dataset))))

    model.eval()
    eval_loss=0
    eval_acc=0

    for data in test_loader:
        img,lable = data
        #img=img.view(img.size(0),-1)
        if use_gpu:
            img=img.cuda()

        out = model(img)
        loss = criterion(out,lable)
        eval_loss+=loss.item()*lable.size(0)
        _,pred=torch.max(out,1)
        num_correct=(pred==lable).sum()
        eval_acc+=num_correct.item()

    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))
    print('Time:{:.1f} s'.format(time.time() - since))
    print()

torch.save(model.state_dict(),'./Neuralnetwork_2.pth')