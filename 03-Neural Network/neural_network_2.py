# -*- coding: utf-8 -*-
"""
Created on Wed May 30 19:31:26 2018

@author: hubinbin
"""

import torch
import torch.nn.functional as F
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import time

batch_size = 32
learning_rate = 1e-3
num_epoches = 10

train_dataset = datasets.MNIST(
    root='./data',train=True,transform=transforms.ToTensor(),download=False)
test_dataset = datasets.MNIST(
    root='./data',train=False,transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

class Neuralnetwork(nn.Module):
    """docstring for Neuralnetwork"""
    def __init__(self, in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(Neuralnetwork, self).__init__()
        self.layer1 = nn.Linear(in_dim,n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1,n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2,out_dim)

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.rrelu(x)
        return x

model = Neuralnetwork(28*28,300,100,10)
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
        img=img.view(img.size(0),-1)
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
        img=img.view(img.size(0),-1)
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