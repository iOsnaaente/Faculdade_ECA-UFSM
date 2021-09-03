# -*- coding: utf-8 -*-
"""PyTorch-Keras_Introdução.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/175xDeWPF-ES0i7uamuwO_4xOa7sKCcXz

Importando os dados de Iris para os treinos
"""

!gdown https://drive.google.com/uc?id=1d3NbjXro_BfnYpFm66ETBfe7ubAZPAoL

from numpy.random import shuffle 

f = open('Iris.csv', 'r')
lines = f.readlines()

X = list()
Y = list()

cats = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

for line in lines[1:]:
  _, sl, sw, pl, pw, sp = line[:-1].split(',')
  
  sl = float(sl)
  sw = float(sw)
  pl = float(pl)
  pw = float(pw)
  
  sp = [ 1.0 if sp == cat else 0.0 for cat in cats ]

  X.append([sl,sw,pl,pw])
  Y.append(sp)


total = len(X)
indexes = list(range(total))

shuffle(indexes)

Xs = [X[i] for i in indexes]
Ys = [Y[i] for i in indexes]

# total = len(X)
sep = int(total * 0.1)

total_train = total - sep
total_test = sep 

Xt = Xs[:total_train,:]
Yt = Ys[:total_train,:]

Xv = Xs[total_train:,:]
Yv = Ys[total_train:,:]

"""Começando a usar PyTorch"""

from matplotlib import pyplot as plt 
import torch

"""Aplicando a descida do gradiente em pyTorch em dados aleatórios"""

x = torch.linspace(0,10, steps=50)
y = (x-5)**2

plt.plot(x,y)
plt.show()

f = lambda x: (x-5)**2 

x_ref = torch.linspace(0,10, steps=50)
y_ref = f(x_ref)

plt.plot(x_ref, y_ref)

x = torch.tensor(1.0, requires_grad=True) #x.requires_grad_(True)

for _ in range(50):
  y = f(x)

  plt.plot(x.data, y.data, 'ro')

  y.backward()
  x.data -= 0.1 * x.grad
  x.grad.zero_()


plt.show()

"""Dados do Iris com PyTorch"""

import torch
import torch.nn as nn 
import torch.nn.functional as F

class Perceptron(nn.Module):
  def __init__(self):
    super(Perceptron, self).__init__()
    self.wb1 = nn.Linear(4, 8)
    self.wb2 = nn.Linear(8, 3)
  
  def forward(self, x):
    s1 = self.wb1(x) 
    z1 = torch.sigmoid(s1)

    s2 = self.wb2(z1)
    z2 = s2              # Calculo do softmax depois
    return z2

p = Perceptron()
print(p)

list(p.parameters())

x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
y = p(x)

y_hat = torch.tensor([[0.0, 1.0, 0.0]])

loss = nn.CrossEntropyLoss()

e = loss(y, y_hat.argmax(dim=1)) # .argmax(dim=1) para pegar o index do onehot 

print(e)

p.zero_grad()
e.backward()

for param in p.parameters():
  param.data -= 0.1 * param.grad

for i in range(10001):
  for x, y_hat in zip(Xt, Xt):
    x = x.view(1,4)
    y_hat = y.view(1,3)
    
    p.zero_grad()
    
    y = p(x)
    
    e = loss(y, y_hat.argmax(dim=1))
    e.backward()
    
    for param in p.parameters():
      param.data -= 0.1*param.grad.data
    
    if not (i%1000) or i == 0:
      print(e)

import torch.optim as optim 

optimizer = optim.SGD(p.parameters(), lr = 0.1)

for i in range(10001):
  optimizer.zero_grad()
  Y = p(Xt)
  e = loss(Y, Yt.argmax(dim=1))
  e.backward()
  optimizer.step()
  if not (i%1000) or i == 0 : 
    print(e)

gpu = torch.device('cuda0') #sla
p.to(gpu)
Xt = Xt.to(gpu)
Yt = Yt.to(gpu)

Xv = Xv.to(gpu)
Yv = Yv.to(gpu)

y = F.softmax(p(Xv), dim = 1)
for y,y_hat in zip(Y,Yv):
  print(y, y_hat)

"""
Keras 
"""

import numpy as np
from tensorflow import keras 
from tensorflow.keras import layers 

#np.set_printoptions(formater = {'float' : lambda x : '%+01.2f' %x})

model = keras.Sequential(
    [
     keras.Input(shape=(4)),
     layers.Dense(8, activation='sigmoid'),
     layers.Dense(3, activation='softmax')
    ]
)

model.summary()

"""Compilar os dados"""

model.compile(loss='categorical_crossentropy', \
              optimizer = 'adam', metrics = ['accuracy']
              )

model.fit(X, Y, batch_size=140, epochs=10001, verbose=False)

