from __future__ import division 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np 
from numpy import linalg
import sympy 


def ajuste(X,Y):

    #Ordem da matriz
    ordem = n[0]+1

    A = np.zeros((ordem,ordem))

    for i in range(ordem):
      for j in range(ordem):
        A[i,j] = sum(X[i]**2n)

    #Matriz aumentada
    A = np.c_[ A, Y ]

    eleminacao_gauss(A)

    coef = np.zeros((n[0]))
    for i in range(n[0]):
        coef[i] = A[i,n[0]] / A[i,i] 

    return(coef)        

def eleminacao_gauss(A):

  #Tamanho
  n = np.shape(x)

  #Zerando elementos abaixo da diagonal principal  
  for i in range(n[0]):
    pivot = A[i,i]
    for j in range(i+1, n[0]):
      m = -A[j,i]/pivot
      for k in range(i, n[0]+1):
        A[j,k] = A[j,k] + m*A[i,k] 

  #Zerando elementos acima da diagonal principal 
  for i in range(n[0]-1,-1,-1):
    pivot = A[i,i]
    for j in range(i-1,-1,-1):
      m = -A[j,i]/pivot
      for k in range(i,n[0]+1):
          A[j,k] = A[j,k] + m*A[i,k]

#Polinomio obtido
def p(r):
  px = 0
  coef = interpolacao(x,y)

  for i in range(n[0]):
    px = coef[i]*r**i + px 
    print(coef[i])
  return px 

#tamanho do vetor
n = np.shape(x)

#grau do polinomio
gr = n[0] - 1

#Exemplos testes
#x = np.array([1,2,3,4],dtype='double')
#y = np.array([-17,4,71,202],dtype='double')

x = np.array(eval(input("Entre com o vetor x: "))) 
y = np.array(eval(input("Entre com o vetor y: "))) 

ajuste = input("O ajuste é polinomial (s/n): ")

if ajuste == 's':

    grau = input("Entre com o grau do polinomio: ")

p = interp1d(x, y, kind='cubic')

xnew = np.linspace(x[0], x[-1], num=20)

plt.plot(x, y, 'o', xnew, p(xnew),'-')
plt.grid()
plt.show()
