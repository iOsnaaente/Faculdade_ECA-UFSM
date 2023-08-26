# -*- coding: utf-8 -*-

import numpy as np
from struct import unpack

def read_imgs(img_filename):
  ''' Esta função lê o arquivo de imagens
      da base de dados MNIST
  '''

  # Abre o arquivo
  img_file = open(img_filename,'rb')

  # Lê o cabeçalho do arquivo
  magic = unpack('>i', img_file.read(4))[0]
  total = unpack('>i', img_file.read(4))[0]
  height = unpack('>i', img_file.read(4))[0]
  width = unpack('>i', img_file.read(4))[0]

  # Verifica se o arquivo passa no teste
  # básico (este número deve ser sempre 2051)
  if magic != 2051:
    print('Erro, este arquivo não parece ser um arquivo de imagens MNIST')

  # Aqui criamos a array do NumPy que armazenará
  # as imagens
  imgs = np.zeros((total,height,width))

  # Nesse laço vamos lendo cada pixel e preenchendo
  # no array
  for k in range(total): # Cada amostra k
    for i in range(height): # Cada linha i
      for j in range(width): # Cada coluna j
        imgs[k,i,j] = ord(img_file.read(1)) # Lemos 1 byte
  
  # Retornamos o array preenchido
  return imgs

"""De forma semelhante ao realizado acima, aqui abaixo definimos as funções auxiliares para leitura do arquivo de rótulos."""

def read_labels(labels_filename):
  ''' Esta função lê o arquivo de rótulos
      da base de dados MNIST
  '''

  # Abre o arquivo
  labels_file = open(labels_filename,'rb')

  # Lê o cabeçalho do arquivo
  magic = unpack('>i', labels_file.read(4))[0]
  total = unpack('>i', labels_file.read(4))[0]

  # Verifica se o arquivo passa no teste
  # básico (este número deve ser sempre 2051)
  if magic != 2049:
    print('Erro, este arquivo não parece ser um arquivo de imagens MNIST')

  # Aqui criamos a array do NumPy que armazenará
  # as imagens
  labels = np.zeros((total))

  # Nesse laço vamos lendo cada label e preenchendo
  # no array
  for k in range(total): # Cada amostra k
    labels[k] = ord(labels_file.read(1)) # Lemos 1 byte
  
  # Retornamos o array preenchido
  return labels

"""Nas linhas abaixo chamamos as função de leitura para carregar as imagens e os respectivos rótulos"""

# Lê dados de treinamento
imgs = read_imgs('train-images-idx3-ubyte')
labels = read_labels('train-labels-idx1-ubyte')

# Lê dados de validação
imgs_val = read_imgs('t10k-images-idx3-ubyte')
labels_val = read_labels('t10k-labels-idx1-ubyte')

print('Imagens de treinamento',imgs.shape)
print('Etiquetas de treinamento', labels.shape)
print('Imagens de validação',imgs_val.shape)
print('Etiquetas de validação', labels_val.shape)


print('labels =',labels,'é um array de',len(labels),'elementos')
print('labels_val =',labels_val,'é um array de',len(labels_val),'elementos')


from matplotlib import pyplot as plt

# No laço abaixo sorteamos três amostras aleatórias
# e mostramos a etiqueta e a respectiva imagem.
for _ in range(3):
  # Sorteamos uma amostra
  i = np.random.randint(0,60000)
  # Imprimimos a etiqueta e a respectiva imagem
  print('A imagem abaixo mostra o dígito', labels[i])
  plt.imshow(imgs[i,:,:],cmap='gray')
  plt.show()


# Escreva aqui seu código para embaralhar
# os pares de treinamento. Mantenha os mesmos
# nomes de variáveis originais.

from random import shuffle

aux = []

for i in range(len(imgs)):
  aux.append((imgs[i], labels[i]))

shuffle(aux)

# Importante manter as dimensões das variáveis usadas 
imgs = np.zeros(imgs.shape)
labels = np.zeros(labels.shape)

for i in range(len(imgs)):
  imgs[i] = aux[i][0]
  labels[i] = aux[i][1]

# Escreva aqui seu código para normalizar
# as imagens dos dados de treinamento, colocando
# os valores no intervalo de 0 a 1

# Se a imagem tiver mais que 8 bits pode ser passado o parametro de bits
norm = lambda pos, bits=8 : pos/((2**bits)-1) 

# Normalização dos valores de imgs
for i in range(len(imgs)):
  for l in range(len(imgs[i])):
    for w in range(len(imgs[i])):
      imgs[i][l][w] = norm(imgs[i][l][w])

# Normalização dos valores de imgs_val
for i in range(len(imgs_val)):
  for l in range(len(imgs_val[i])):
    for w in range(len(imgs_val[i])):
      imgs_val[i][l][w] = norm(imgs_val[i][l][w])


# Escreva aqui o código que converte os
# arrays labels e labels_val para o formato
# one-hot

ZERO = [0,0,0,0,0,0,0,0,0,0]

lab = np.zeros( (len(labels), 10) )
lab_val = np.zeros( (len(labels_val), 10) )

def oneHot(pos):
  aux = ZERO.copy()
  aux[int(pos)] = 1
  return aux

for i in range(len(lab)):
  lab[i] = oneHot( labels[i] )

for j in range(len(lab_val)):
  lab_val[j] = oneHot(labels_val[j])

labels = lab.copy()
labels_val = lab_val.copy()


# Implemente aqui a função softmax

from numpy import exp

def softmax(x):  
  return exp(x)/np.sum(exp(x))


# Implemente aqui a função sigmoide

def sigmoid(x):
  return np.array([1/(1+exp(-i)) for i in x])


# Implemente aqui, passo a passo, a classe de sua rede neural

class Perceptron():

  def __init__(self):
  
    # Pesos (width) 
    self.W1 = np.random.random((256,784)) * 2 - 1
    self.W2 = np.random.random((64, 256)) * 2 - 1
    self.W3 = np.random.random((10, 64 )) * 2 - 1
    
    #Bias
    self.b1 = np.random.random((256,1)) * 2 - 1
    self.b2 = np.random.random((64,1)) * 2 - 1
    self.b3 = np.random.random((10 ,1)) * 2 - 1 
    
    # Passo
    self.eta = 0.001
  
  def forward(self, inputs):

    # Garante vetor coluna
    inputs = np.reshape(inputs, (len(inputs),1))
    
    # Sinapse Hidden 1 camada = Entrada * Pesos Input w1 + bias b1
    self.s1 = np.dot(self.W1, inputs) + self.b1
    self.z1 = sigmoid(self.s1)

    # Sinapse Hidden 2 camada = saida z1 * Pesos w2 + bias b2
    self.s2 = np.dot(self.W2, self.z1) + self.b2
    self.z2 = sigmoid(self.s2)

    # Sinapse Output = Saida z2 * Pesos w3 + bias b3
    self.s3 = np.dot(self.W3, self.z2) + self.b3
    self.z3 = softmax(self.s3)
    
    return self.z3

  
  # Implementação do backpropagation 
  def backprop(self, X, Y_des ):
    
    self.y = self.forward(X)

    # Delta da camada de saida usando Cross Entropy 
    # δL = y - y_des
    self.d3 = self.y - Y_des

    # δl=W(l+1).T *δ(l+1) ⊙ σ′(sl)
    # σ′(sl) = zl(1−zl)
    
    self.d2 = np.dot(self.W3.T, self.d3) * self.z2 *(1-self.z2)
    self.d1 = np.dot(self.W2.T, self.d2) * self.z1 *(1-self.z1)


    # Calculo das derivadas parciais de W[n.m] e B[n]
    self.dW3 = np.dot(self.d3, self.z2.T)
    self.db3 = self.d3

    self.dW2 = np.dot(self.d2, self.z1.T)
    self.db2 = self.d2

    self.dW1 = np.dot(self.d1, X.T)
    self.db1 = self.d1


    # Passo 
    self.eta = 0.1 

    # Optimização dos pesos e biases
    self.W1 = self.W1 - self.eta * self.dW1  
    self.W2 = self.W2 - self.eta * self.dW2  
    self.W3 = self.W3 - self.eta * self.dW3  
    
    self.b1 = self.b1 - self.eta * self.db1 
    self.b2 = self.b2 - self.eta * self.db2
    self.b3 = self.b3 - self.eta * self.db3

    # Cross Entropy 
    self.ce = -np.sum(Y_des * np.log(self.y))

    return self.ce


def train_batch(p, X, Y_desired, batch_size=250):
    ''' Esta função faz o treinamento da rede
        neural, percorrendo todo dataset, por
        lotes de 250 amostras

        PS.: Aqui os lotes não importam muito
             pois ajustamos os pesos um pouco
             a para cada amostra individual.
    '''

    # Total de amostras
    total = X.shape[0]

    # Erro global vai ser somado aqui
    Err = 0.0

    # Vamos percorrer as amostras em lotes
    for i in range(0,total,batch_size):

      # Erro de cada lote
      err_batch = 0.0

      # Aqui neste laço vamos treinar o lote
      for j in range(i,i+batch_size):

        # Separamos os dados de entrada
        x = np.reshape(X[j,:,:],(784,1))

        # Separamos os dados de treinamento correspondentes
        y_desired = np.reshape(Y_desired[j,:],(10,1))

        # Calculamos o fator de correção dos pesos e biases
        ce = p.backprop(x, y_desired)

        # Computamos o erro do lote
        err_batch += ce

      # Normalização do erro do lote
      err_batch /= batch_size

      # Soma do erro do lote ao erro global
      # já com fator de normalização
      Err += err_batch / (total/batch_size)

    return Err


import time

# Aqui criamos a rede neural
p = Perceptron()

# Nesta lista gravaremos a evolução do erro
# para plotar num gráfico mais tarde
Errs = []

# Treinaremos 10 épocas (cada época demora em torno de 2 min)
for i in range(10):

  # Marcamos o tempo de início para computar o tempo
  # que demoramos para treinar cada época (isso ajuda
  # a estimar o tempo total)
  start_time = time.time()

  # Aqui fazemos o treinamento
  Err = train_batch(p, imgs, labels)

  # Mostramos os resultados parciais na tela
  print('Elapsed time:', time.time() - start_time, 's', \
        'Err:', Err)
  
  # Guardamos o erro calculado em cada época para
  # plotar no gráfico em seguida
  Errs.append(Err)

"""# Avaliação dos Resultados"""

# Plotamos o gráfico da evolução do erro
# no tempo. Esta é a chamada "curva de
# aprendizagem"

plt.plot(Errs)
plt.show()

"""A função abaixo serve para calcular a taxa de acerto dessa rede neural"""

def accuracy(p, X, Y):
  ''' Esta função vai calcular a taxa
      de acerto da rede neural p
      nos dados fornecidos
  '''

  # Contador de acertos
  correct_count = 0

  # Total de amostras
  total = X.shape[0]

  # Laço vai percorrer todas amostras
  for k in range(total):

    # Esta é a resposta que desejamos
    correct_answer = np.argmax(Y[k,:])

    # Esta é a resposta encontrada
    guess = np.argmax(p.forward(np.reshape(X[k,:,:],(784,1))))

    # Se ambas estiverem corretas
    if correct_answer == guess:

      # Contabilizamos como resposta correta
      correct_count += 1

  # Aqui retornamos o resultado
  return correct_count / total

print('Taxa de acerto nos dados de treinamento:', \
      100*accuracy(p, imgs, labels), '%')

print('Taxa de acerto nos dados de validação:', \
      100*accuracy(p, imgs_val, labels_val), '%')