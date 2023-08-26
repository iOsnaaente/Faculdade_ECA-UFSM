import matplotlib.pyplot as plt
from random import randint
from sympy import symbols
import numpy as np

from math import *

# Primeiro definimos a variável que será lida f(x) = x 
x = symbols('x')

# Função dos mínimos quadrados
def minimos(x,y, fun, val):

    q = len(fun)

    # Cálculo de A
    A = np.zeros((q,q), dtype='float')
    for i in range(q):
        for j in range(q):
            A[i][j] = 0 
            for k in range(len(x)):
                A[i][j] = A[i][j] + fun[i](x[k])*fun[j](x[k])
    
    # Cálculo de B 
    b = np.zeros(q, dtype='float')
    for i in range(q):
        b[i] = 0 
        for j in range(len(x)):
            b[i] = b[i] + y[k]*fun[i](x[j])
    
    # Encontrar A.a = b -> a = b.inv(A)
    a = np.linalg.inv(A) *b 
    
    # Cálcular Yt = fun[i](val)
    yt = 0
    for i in range(q):
        yt = yt + a[i]*fun[i](val)
    
    # Valor do mínimo 
    return yt 


if __name__ == "__main__":    

    # INSTRUÇÕES
    print(' #########################################################')
    print(' ###             EXEMPLO DE ENTRADA:                   ###')
    print(' ###       x**2 + 5*x - 4  ou    cos(x)* x**2          ###') 
    print(' #########################################################\n')

    '''
    # Entrada da função escolhida para o método
    print("Dê entrada com funções F1(x) , F2(x) , ... , Fn(x):")
    nomes = input('\n').split('/')

    funcoes = [lambda x: eval(i) for i in range(len(nomes))]
    '''

    funcao1 = 'x**2'
    funcao2 = 'x'
    funcao3 = '1'

    nomes = [funcao1, funcao2, funcao3]
    nomes = [ nomes[i].replace('**','^') for i in range(len(nomes))]

    # Aplica às funções o valor de x dado de entrada
    f1 = lambda x: eval(funcao1)     
    f2 = lambda x: eval(funcao2)    
    f3 = lambda x: eval(funcao3) 

    funcoes = [f1,f2,f3]
    
    #x  = map(float, input("Entre com o vetor de X: ").split(' '))
    #y  = map(float, input("Entre com o vetor de Y: ").slipt(' '))

    x = '-1; -0.75; -0.6; -0.5; -0.3; 0; 0.2; 0.4; 0.5; 0.7; 1'
    y = '2.05; 1.153; 0.45; 0.40; 0.5; 0; 0.2; 0.6; 0.512; 1.2; 2.05'
    
    x = (x.split(';'))
    y = (y.split(';')) 

    x = [float(x[i]) for i in range(len(x))]
    y = [float(y[i]) for i in range(len(y))]
 
    #xt = input("\nDê entrada a um valor de X : ")
    xt = 0.6 

    p = minimos(x, y, funcoes, xt )
    print(p)

    plt.ylabel('f(x)')
    plt.xlabel('x')
    plt.title('Método dos mínimos quadrados')
    
    xfuncMMQ = []
    yfuncMMQ = []
    for fx in np.linspace(round(x[0]),round(x[-1]), round((abs(x[0])+abs(x[-1])*100)) ):
        xfuncMMQ.append(fx)
        yfuncMMQ.append(p[2]*fx**2 + p[1]*fx + p[0])
        
    plt.plot(x , y , 'o',label = 'Pontos definidos') 
    plt.plot(xfuncMMQ , yfuncMMQ , '-',label = 'Função aproximada %5.4fx² %5.4fx %5.4f' %(p[2],p[1],p[0])) 
    
    plt.legend(fancybox = True)

    plt.show()
