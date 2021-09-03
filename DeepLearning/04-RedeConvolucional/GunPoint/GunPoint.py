# -*- coding: utf-8 -*-

# Importa as bibliotecas que usaremos

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

import numpy as np
from matplotlib import pyplot as plt

# Ajusta os parâmetro de precisão para
# as variáveis que serão impressas na tela

torch.set_printoptions(precision=2, sci_mode=False)
np.set_printoptions(precision=2)


def load_file(filename):
  ''' Essa função lê o arquivo com a série de
      dados do dataset GunPoint, retornando
      o par de entrada e saída desejada
  '''

  # Essas listas vão conter as entradas (séries temporais)
  # e respectivas saídas desejadas (classes no formato
  # one-hot)

  X = list()
  Y = list()

  # Abre o arquivo e carrega as linhas

  f = open(filename, 'r')
  lines = f.readlines()

  # Laço que converte os dados lidos,
  # linha por linha

  for line in lines:

    # Aqui lemos a sequência de dados ponto flutuante,
    # separada por espaços duplos, descartando os espaços
    # iniciais e o caracter de nova linha ao final de
    # cada linha

    data = [float(x) for x in line[3:-1].split('  ')]

    # O primeiro número representa a classe que
    # aqui é convertida o índice zero ou um

    Y.append(int(data[0]-1))

    # Os demais números são a sequência de posições
    # da mão do ator ou da atriz no eixo x

    X.append(data[1:])
  
  # Embaralhamento das amostras

  idxs = list(range(len(X)))
  np.random.shuffle(idxs)
  X_ = list()
  Y_ = list()
  for i in idxs:
    X_.append(X[i])
    Y_.append(Y[i])

  # Retornamos o par X, Y
  
  return torch.tensor(X_), torch.tensor(Y_)

# Leitura dos pares de treinamento e validação

Xt, Yt = load_file('GunPoint_TRAIN.txt')
Xv, Yv = load_file('GunPoint_TEST.txt')


# Aqui verificamos o tamanho do dataset
# e a quantidade de amostras em cada série

print(Xt.size())
print(Xv.size())

# Aqui plotamos alguns exemplos de dados
# desse dadaset para examinar como são
# as curvas de cada classe.

fig, ax = plt.subplots(1, 1, figsize=(8,5))
ax.set_ylim(-3, 3)
ax.set_title('GunPoint Dataset')
for _ in range(30):
  idx = np.random.randint(0,len(Xt))
  c = Yt[idx]
  ax.plot(Xt[idx], 'r' if c == 0 else 'b')
plt.show()


class ConvNet(nn.Module):
  ''' Essa é nossa classe da rede
      convolucional que estamos criando
      manualmente.
  '''
  def __init__(self):

    # Construtor da classe mãe
    super(ConvNet, self).__init__()

    # A janela da convolução terá tamanho 10
    # e uma saída apenas (se diz que tem 1 canal
    # de saída)

    self.wb1 = nn.Linear(20,1)

    # Depois da convolução adicionamos apenas
    # mais uma camada que coleta os 29 valores
    # gerados durante a convolução anterior e
    # e calcula duas saídas

    self.wb2 = nn.Linear(14,2)

  def forward(self, x):

    # Aqui criamos o tensor que armazenará
    # os resultados da convolução
    s1 = torch.zeros(len(x),14)
    
    # Se estivermos usando a GPU vamos
    # então mover essa variável para lá
    if torch.cuda.is_available():
      s1 = s1.cuda()

    # Esse laço aplica a convolução
    for i in range(14):
      s1[:,i:i+1] = self.wb1(x[:,i*10:i*10+20])
    
    # Ativação
    z1 = torch.sigmoid(s1)

    # Última camada e valor de retorno
    s2 = self.wb2(z1)
    return s2

# Aqui criamos o objeto

cnn = ConvNet()
print(cnn)

print(list(cnn.parameters()))


# Aqui criamos o otimizador e a função
# de perda

opt = optim.SGD(cnn.parameters(), lr=1.0)
loss = nn.CrossEntropyLoss()

# Movemos tudo para a GPU
# (essa parte é opcional)

gpu = torch.device("cuda:0")
cnn = cnn.to(gpu)
Xt = Xt.to(gpu)
Yt = Yt.to(gpu)
Xv = Xv.to(gpu)
Yv = Yv.to(gpu)

# Treinamento por 10 mil épocas
# (pode repetir essa célula várias
#  vezes para tentar aumentar a
#  acurácia)

for j in range(10001):
  x = Xt
  y_hat = Yt
  opt.zero_grad()
  y = cnn(x)
  e = loss(y, y_hat)
  e.backward()
  opt.step()
  if not (j % 1000):
    print(float(e))
    

# Aqui verificamos a acurácia nos dados
# de validação

x = Xv
y_hat = Yv
y = cnn(x).argmax(dim=1)
print('Acurácia',100*float((y == y_hat).sum()) / len(x))