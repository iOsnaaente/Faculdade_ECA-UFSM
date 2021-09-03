from __future__ import division 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np 
from numpy import linalg
import sympy 

def newton(X,Y,x):
    for j in range(1,n[0]):
        aux = 0 
        for i in range(n[0]-j):
            num = A[i+1,j-1] - A[i,j-1]
            den = X[i+j] - X[aux] 
            A[i,j] = round(num/den,10)
            aux = aux + 1

    p = np.zeros((n[0]))
    for i in range(n[0]):    
        p[i] = A[0,i]

    L = np.ones(n[0])
   
    for i in range(n[0]-1):
        aux = 0
        for j in range(i+1):
                aux = (x-X[j]) 
                L[i+1] = L[i+1]*aux
    px = 0
    sun = 0
    for i in range(n[0]):
        sun = p[i]*L[i]
        px = px + sun
    return px

#Polinomio de newton
px = lambda x: newton(X,Y,x)

#Exemplos testes
#X = np.array([0,0.5,1,1.5,2,2.5],dtype='double')  
#Y = np.array([-2.78,-2.241,-1.65,-0.594,1.34,4.564])

#X = np.array([1,1.3,1.6,1.9,2.2],dtype='double')  
#Y = np.array([0.7651977,0.6200860,0.4554022,0.2818186,0.1103623],dtype='double')

#Entre com os  vetores
a = input("Entre com o vetor x ex(1 2 3): ")
b = input("Entre com o vetor y ex(2 4 8): ")

X = np.array([float(i) for i in a.split()])
Y = np.array([float(i) for i in b.split()])

#tamanho do vetor 
n = np.shape(X)
A = np.zeros((n[0],n[0]))

#Preencher toda coluna 0 por Y
A[:,0] = Y 

#chamar a func
px(2)

#Exemplo teste
valor = eval(input("Valor do polinomio de interpolação no ponto px: "))
print(px(valor))

p = interp1d(X, Y, kind='cubic')

xnew = np.linspace(X[0], X[-1], num=20)

plt.plot(X, Y, 'o', xnew, p(xnew),'-')
#plt.plot(px(valor), 'o', xnew)
plt.grid()
plt.show()

