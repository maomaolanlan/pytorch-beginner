# -*- coding: utf-8 -*-
"""
Created on Fri May 11 21:12:49 2018

@author: hubinbin
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# make fake data
n_data = torch.ones(100,2)
x0 = torch.normal(2*n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(100)

x = torch.cat((x0,x1), 0).type(torch.FloatTensor)
y = torch.cat((y0,y1),).type(torch.LongTensor)

plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.show()

plt.ion()

class Net(torch.nn.Module):
    """docstring for Net"""
    def __init__(self, n_feature, n_hidden, n_outpul):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.out = torch.nn.Linear(n_hidden,n_outpul)
    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x

net = Net(n_feature=2,n_hidden=10,n_outpul=2)
# print(net)
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()

num_epochs = 100
for epoch in range(num_epochs):
    out = net(x)
    loss = loss_func(out,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1)%5 == 0:
        print('epoch[{}/{}],loss:{:.6f}'.format(epoch+1, num_epochs,loss.data[0]))
        plt.cla()
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=pred_y, s=100, lw=0,cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200
        plt.text(1.5,-4,'Accuracy=%.2f'% accuracy,fontdict={'size':20,'color':'red'})
        plt.pause(1)
plt.ioff()
plt.show()