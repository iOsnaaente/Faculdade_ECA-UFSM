# -*- coding: utf-8 -*-

import numpy as np

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from matplotlib import pyplot as plt

# Leitura dos dados
X = np.load('X.npy')
Y = np.load('Y.npy')

# Reordena as categorias na ordem correta
# (por motivo que desconheço, os dados
#  originais estavam com as classes fora
#  de ordem -- consistentes e organizadas,
#  mas fora de ordem)
cats = [9,0,7,6,1,8,4,3,2,5]
Y[:,cats] = Y[:,range(10)]


def split_and_shuffle(X, Y, perc = 0.1):
  ''' Esta função embaralha os pares de entradas
      e saídas desejadas, e separa os dados de
      treinamento e validação
  '''
  # Total de amostras
  tot = len(X)
  # Emabaralhamento dos índices
  indexes = np.arange(tot)
  np.random.shuffle(indexes)
  # Calculo da quantidade de amostras de
  # treinamento
  n = int((1 - perc)*tot)
  Xt = X[indexes[:n]]
  Yt = Y[indexes[:n]]
  Xv = X[indexes[n:]]
  Yv = Y[indexes[n:]]
  return Xt, Yt, Xv, Yv

# Aqui efetivamente realizamos a separação
# e embaralhamento

Xt, Yt, Xv, Yv = split_and_shuffle(X, Y)

# Transforma os arrays do NumPy em
# tensores do PyTorch

Xt = torch.from_numpy(Xt)
Yt = torch.from_numpy(Yt)
Xv = torch.from_numpy(Xv)
Yv = torch.from_numpy(Yv)

# Adiciona dimensão dos canais
# (único canal, imagem monocromática)

Xt = Xt.unsqueeze(1)
Xv = Xv.unsqueeze(1)

print('Dados de treinamento:')
print('Xt', Xt.size(), 'Yt', Yt.size())
print()
print('Dados de validação:')
print('Xv', Xv.size(), 'Yv', Yv.size())


def show_sample(X, Y, n=3):
  ''' Essa função mostra algumas
      amostras aleatórias
  '''
  for i in range(n):
    k = np.random.randint(0,len(X))
    print('Mostrando', int(torch.argmax(Y[k,:])))
    plt.imshow(X[k,0,:,:], cmap='gray')
    plt.show()

show_sample(Xt, Yt)

# Para cada uma das variáveis abaixo
# substitua None pelo valor inteiro
# correto.

N1 = 5        # Valor do número de canais 
N2 = 30       # Linhas e colunas wo = (64 + 2.0 + 6)/2 + 1
N3 = 4500     # Canais*Linhas*Colunas 

N4 = 5        # Canais não mudam com Pool
N5 = 15       # 2x2 Stride 2 reduz pela metade 
N6 = 1125     # Canais*Linhas*Colunas 

N7 = 8        # Número de canais da saída da conv2 
N8 = 13       # Colunas e linhas
N9 = 1352     # Canais*Colunas*Linhas

N10 = 8       # Canal não muda com Pool
N11 = 6       # Metade para cima de 13

N12 = 288     # Saídas Coluna*Linhas*Canais

N13 = 10      # Saída de dados sum(0 -> 9) = 10


# Escreva aqui o código da classe que
# implementará sua rede neural

class ConvNet(nn.Module):
  
  def __init__(self):
    super(ConvNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 5, 6, 2)  # Canais entrada, saída, kernel, stride
    self.pool1 = nn.MaxPool2d(2,2)      # Kernel, stride
    self.conv2 = nn.Conv2d(5, 8, 3, 1)  # Canais entrada, saída, kernel, stride
    self.drp1  = nn.Dropout2d(0.25)     # Percentagem de morte
    self.pool2 = nn.MaxPool2d(2,2)      # Kernel, stride 
    self.lin1  = nn.Linear(288, 10)     # Colunas*Linhas*Canais, saída 10 digitos 

  def forward(self, x):
    x = self.conv1(x)
    x = torch.relu(x)
    x = self.pool1(x)
    x = self.conv2(x)
    x = self.drp1(x)
    x = torch.relu(x)
    x = self.pool2(x)
    x = x.view(-1, 288)
    x = self.lin1(x)
    return x

cnn = ConvNet()
print(cnn)


def evaluate(x, y_hat):
  ''' Calcula a acurácia da ConvNet (variável cnn)
      para o par de entradas e saídas desejadas
      x, y_hat. Aqui assume-se que y_hat está
      originalmente no formato one-hot. Tanto
      x quanto y_hat devem ser lotes, não amostras
      individuais.
  '''
  y = cnn(x).argmax(dim=1)
  y_hat = y_hat.argmax(dim=1)
  return 100*float((y == y_hat).sum()) / len(y)


opt = optim.Adam(cnn.parameters(), lr=0.0001)
loss = nn.CrossEntropyLoss()


# Movemos tudo para a GPU
# (essa parte é opcional)

gpu = torch.device("cuda:0")
cnn = cnn.to(gpu)
Xt = Xt.to(gpu, dtype=torch.float)
Yt = Yt.to(gpu, dtype=torch.long)
Xv = Xv.to(gpu, dtype=torch.float)
Yv = Yv.to(gpu, dtype=torch.long)

# Laço de treinamento para 2001
# épocas

for j in range(2001):

  # Faremos o treinamento em lotes de
  # tamanho igual a 128 amostras

  for i in range(0,len(Yt),128):

    # Separa o lote de entradas
    x = Xt[i:i+128,:,:,:]

    # Separa o lote de saídas desejadas
    # já transformando de one-hot para
    # índice das colunas.
    y_hat = Yt[i:i+128,:].argmax(dim=1)

    # Zera o gradiente do otimizador
    cnn.zero_grad()

    # Calcula a saída da rede neural
    y = cnn(x)

    # Calcula o erro
    e = loss(y, y_hat)

    # Calcula o gradiente usando backpropagation
    e.backward()

    # Realiza um passo de atualização dos parâmetros da rede neural usando o otimizador.
    opt.step()

  # A cada 200 épocas imprimimos o
  # erro do último lote e a acurácia
  # nos dados de treinamento
  if not (j % 200):
    print(float(e), evaluate(Xt, Yt))


cnn.eval() # desliga dropout

# Não modifique essa célula.

ac = evaluate(Xv, Yv)
print('Acurácia de', ac, '%')


def random_sample_cnn(X, Y):
  ''' Essa função testa a rede convolucional
      mostrando a imagem de entrada, a saída
      calculada, e a saída esperada, para
      5 amostras aleatórias.
  '''
  for _ in range(5):
    idx = np.random.randint(0, len(Yv))
    x = Xv[idx:idx+1,:,:,:]
    y = int(cnn(x).argmax(dim=1))
    y_hat = int(Yv[idx:idx+1,:].argmax(dim=1))
    print('y =', y, 'y_hat =', y_hat)
    x = x.cpu()
    plt.imshow(x[0,0,:,:], cmap='gray')
    plt.show()

# Aqui examinamos alguns exemplos
# aleatórios nos dados de validação

random_sample_cnn(Xv, Yv)