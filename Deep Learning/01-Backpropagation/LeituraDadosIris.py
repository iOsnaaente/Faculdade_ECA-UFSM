# -*- coding: utf-8 -*-

# Datashet Iris Flower Datashet
# 3 tipos de Iris (Iris Virginica - Iris Setosa - Iris Versicolor)

# Permite o Google fazer upload 
from google.colab import files 
import io

# Faz o upload do arquivo csv
upFile = files.upload()

"""**2) Lê todas as linhas**"""

# Executa o binário do arquivo csv
File = io.BytesIO(upFile["Iris.csv"])

# Mantem o ponteiro de leitura no primeiro byte
File.seek(0)
# Lê as linhas do arquivo ('\n')
Lines = File.readlines()

import numpy as np 

x = np.zeros((len(Lines)-1, 4), dtype='float')
y = np.zeros((len(Lines)-1,3), dtype='float')

categorias = np.array(("Iris-setosa", 'Iris-versicolor', 'Iris-virginica'))

"""**3) Remove o cabeçalho (Primeira linha)**"""

# Guarda as informações do cabeçalho antes de remover
header = Lines[0][:-1].decode().split(',')
# Remove a linha 0 (cabeçalho)
Lines.remove(Lines[0])

"""**4) Mostra o número total de linhas**"""

print("O número total de linhas de dados é ", len(Lines) )

print(header)

"""**5)Decodifica as linhas para string simples?	
6)Remove '\n' do final de cada linha?	
7)Separa os dados de cada linha usando vírgula como espaçador?	
8)Cria uma array do NumPy com todos dados de entrada no formato ponto flutuante?
9)Cria outra array do NumPy com as saídas no formato one-hot?**
"""

for id, line in enumerate(Lines[1:]):

  # underscore ignora os elementos dessa posição
  _, sl, sw, pl, pw, s = line[:-1].decode().split(",")
  
  # Converte os elementos lidos para float
  sl = float(sl)
  sw = float(sw)
  
  pl = float(pl)
  pw = float(pw)

  # Monta um array com os valores de entrada em X
  x[id:] = np.array([sl,sw,pl,pw])
  # Monta os valores de saída no estilo One-Hot
  y[id:] = (categorias == s).astype('float')

  print(x[id],y[id])

"""**9)Encontra o maior e o menor valor para cada coluna do array de entradas?**"""

maxCol = [ 0 for num in range(len(x[0]))]
minCol = [ 0 for num in range(len(x[0]))]

for i in range(len(x[0])):
  # Minimo e Máximo obtidos com a função Min Max da transposta de X
  maxCol[i] = max(x.T[i])
  minCol[i] = min(x.T[i])
  print("Máximo coluna %s : %s \nMinimo coluna %s : %s\n" %(i, maxCol[i], i, minCol[i]) )

"""**10)Normaliza o array de entradas para que todos valores estejam no intervalo de 0 a 1?** 

**11)O valor mínimo corresponde ao zero e o valor máximo corresponde a 1 para cada coluna?**
"""

normalizado = []

for i in range(4):
  # Utiliza o método de normalização de um elemento
  normalizado.append((x.T[i] - minCol[i]) / (maxCol[i]-minCol[i]))
print(normalizado)

"""
**Embaralha as linhas das entradas e das saídas, mantendo os pares correspondentes?**"""

from random import shuffle 

dataArray = []

for i in range(len(x)):
  # Junta os dados em uma só array no estilo [(entrada), (saída)]
  dataArray.append( (x[i], y[i]) )

# Embaralha usando o método shuffle do random
shuffle(dataArray)

#print(dataArray[0][0], dataArray[0][1])
print(dataArray)

"""**Separa dados de treinamento e validação na proporção 90%/10%?** """

from random import randint 

treinamento = []
validacao = []

for data in dataArray:
  if randint(0,100) > 90: 
    treinamento.append(data[0])
  else: 
    validacao.append(data)

print(len(validacao), len(treinamento))

print(len(validacao)*100/150, len(treinamento)*100/150)