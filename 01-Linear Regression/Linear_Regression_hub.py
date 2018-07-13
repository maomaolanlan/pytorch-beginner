# -*- coding: utf-8 -*-
"""
Created on Fri May 11 18:06:45 2018

@author: hubinbin
"""
__author__ = 'copy_dog'

import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([[3.3],[4.5],[5.5],[6.71],[6.93],[4.168],
                    [9.779],[6.182],[7.59],[2.167],[7.4042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

class LinearRegression(nn.Module):
    """docstring for LinearRegression"""
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1,1) # y=w*x+b

    def forward(self,x):
        out = self.linear(x);
        return out

model = LinearRegression();
# 误差
criterion = nn.MSELoss()
# 参数优化函数
optimizer = optim.SGD(model.parameters(),lr=1e-4)

# Train
num_epochs = 1000
for epoch in range(num_epochs):
    # forward
    out = model(x_train)
    loss = criterion(out, y_train)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1)% 20 == 0:
        print('Epoch[{}/{}],loss:{:.6f}'.format(epoch+1, num_epochs,loss.data[0]))

model.eval()
predict = model(x_train)
predict = predict.data.numpy()

plt.plot(x_train.numpy(),y_train.numpy(),'ro',label='Original data')
plt.plot(x_train.numpy(),predict,label='Fitting Line')

plt.legend()
plt.show()

torch.save(model.state_dict(), './linear.pth')
