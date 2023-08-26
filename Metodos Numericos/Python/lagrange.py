from __future__ import division 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np 
from numpy import linalg
import sympy 

def lagrange(X,Y,x):
    L = np.ones(n[0])
   
    for i in range(n[0]):
        aux = 0
        for j in range(n[0]):
            if (i != j):
                aux = (x-X[j])/(X[i]-X[j]) 
                L[i] = L[i]*aux
    px = 0
    sun = 0
    for i in range(n[0]):
        sun = Y[i]*L[i]
        px = px + sun
    return px

#Polinomio de lagrange

px = lambda x: lagrange(X,Y,x)

#Exemplos testes
#X = np.array([1,2,3,4],dtype='double')
#Y = np.array([-17,4,71,202],dtype='double')


#Nesse exemplo em diante deixei de usar a função eval()
#por não ser a mais indicada nesse uso, visto que ela transforma tudo que
#for digitado em comando, o cast float() é o ideal.

a = input("Entre com o vetor x ex(1 2 3): ") 
b = input("Entre com o vetor y ex(5 1 5): ") 


X = np.array([float(i) for i in a.split()])
Y = np.array([float(i) for i in b.split()])

n = np.shape(X)

#grau do polinomio
gr = n[0] - 1

#Exemplo teste
valor = eval(input("Valor do polinomio de interpolação no ponto px: "))
print(px(valor))

p = interp1d(X, Y, kind='cubic')

xnew = np.linspace(X[0], X[-1], num=20)

plt.plot(X, Y, 'o', xnew, p(xnew),'-')
plt.grid()
plt.show()
