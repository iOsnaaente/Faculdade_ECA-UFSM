import numpy as np 
from itertools import *

x = np.array([100,121,1447],dtype='double')
y = np.array([10,11,12],dtype='double')

#Usando regra de cramer p/ encontrar os coeficientes a0,a1,a2,a3
#do polino de interpolação dos pontos acima 

def coeficientes():
  matriz = np.zeros((3,3)) 
  for i in range(3):  
    for j in range(3):
      matriz[i][j] = x[i]**(0+j)

  print(matriz)
  print("\n")
  detA = np.linalg.det(matriz)
  print("\n")

#Vetor dos coeficientes do polinomio 
  a = np.zeros((3))

  for j in range(3):
    for i in range(3):
      matriz[i][j] = y[i]
      
    a[j] = (np.linalg.det(matriz))/detA

    for u in range(3):  
      for z in range(3):
        matriz[u][z] = x[u]**(z)
  
  print(matriz)
  print('\n')
  
  
  return a 

a = coeficientes()

print(a)

#Polinomio de interpolação

def polinomio(x):
  return a[0]+a[1]*(x)+a[2]*(x**2)

print("\nPolinomio de interpolação e^x, pela definição, calculado em 3.1 = %.15f, Error %.15f" % (polinomio(115),np.sqrt(115)-polinomio(115)))

##################################################
###### Caluculo polinomio de lagrange ############
##################################################

#def lagrange(x, fx):                                                             
#    L = lambda num, xi: product((num - xj) / (xi - xj) for xj in x if xj != xi)                                                                                   
#    return lambda num: sum([yi * L(num, xi) for xi, yi in zip(x, fx)])

#A = lagrange(x,y)

X = np.array([0.1,0.3,0.5,0.7],dtype='double')
Y = np.array([-0.80484,-0.44082,-0.10653,0.20341],dtype='double')

def calculaP(x):
    valor = 0
    for k in range(4):
        lk = 1.0
        for i in range(4):
            if k != i:
                lk = lk*(x - X[i])/(X[k]-X[i])
        valor = valor + Y[k]*lk 
    return valor

print()
print("\nPolinomio de interpolação e^x, por lagrange, calculado em 3.1 = %.15f, Error %.15f" % (calculaP(0),np.e**(3.1)-calculaP(3.1)))



def newton(x):
  return 11.02+13.55*(x-2.4)+8.34375*(x-2.8)*(x-2.4)+3.38541667*(x-2.8)*(x-2.4)*(x-3.2)

print("\nPolinomio de interpolação e^x, newton, calculado em 3.1 = %.15f, Error %.15f" % (newton(3.1),np.e**(3.1)-newton(3.1)))


