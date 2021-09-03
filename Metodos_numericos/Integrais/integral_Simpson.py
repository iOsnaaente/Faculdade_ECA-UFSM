from numpy import linspace
from sympy import symbols

from math import *

# Definição da variável em uso -> f(x): x
x = symbols('x')
    
def Simpson(f, a, b, e):

    #Um numero muito pequeno
    d = 0.0001

    #Vetor de m pontos cobrindo o intervalo [a,b] 
    xd = []
    yd = []

    for i in linspace(a,b,round((b-a)/d) ):
        xd.append(i)
        yd.append((f(i+4*d) - 4*f(i+3*d) + 6*f(i+2*d) - 4*f(i+d) +  f(i) )/(d**4))

    #Valor máximo da derivada segunda dos subintervalos 
    Max = max(yd)
    Min = min(yd)
    M = abs(Max) if abs(Max)>abs(Min) else abs(Min) 

    #Calculo de m 
    m = ( ( ( (b-a)**5 )*M ) / (180*e))**(1/4)

    #O arredondamento deve ser feito para cima
    if m > round(m): 
        m = round(m)+1
    else:
        m = round(m)

    #Deve-se tomar o primeiro número par, maior que m 
    if m%2 is 1: 
        m = m+1

    h = (b-a)/(m)
    s = f(a) + f(b)

    x = a +h

    for i in range(0,m-1,2):
        s = s + 4*f(x)
        x = x + 2*h

    x = a +2*h
    for i in range(1, m-2, 2):
        s = s + 2*f(x)
        x = x + 2*h
    
    Isr = (h/3)*s

    return Isr

if __name__ == "__main__":
    # INSTRUÇÕES
    print(' #########################################################')
    print(' ###             EXEMPLO DE ENTRADA:                   ###')
    print(' ###       x**2 + 5*x - 4  ou    cos(x)* x**2          ###') 
    print(' #########################################################\n')

    
    # Entrada da função escolhida para o método
    '''
    funcao = input("Entre com uma funcao f(x): ")

    A = float(input("Entre com o ponto A: "))
    B = float(input("Entre com o ponto B: "))
    e = float(input("Entre com o erro e: "))
    '''

    funcao = 'x**2 + 1'
    A = 0 
    B = 2
    e = 0.001

    f = lambda x: eval(funcao) 

    val = Simpson(f, A, B, e)

    try:
        import matplotlib.pyplot as plt
        matplot = True

    except: 
        print("MatplotLib indisponível!!")
        matplot = False
        
    if matplot:
        _sum = (abs(A)+abs(B))
        
        x = []
        y = []
        for dot in linspace(A-(1/4)*_sum, B+(1/4)*_sum, _sum*100 ):
            x.append(dot)
            y.append(f(dot))

        x_b = []
        y_b = []
        for i in linspace(A, B, _sum*100 ):
            x_b.append(i)
            y_b.append(f(i))

        plt.grid = True

        plt.ylabel('f(x)')
        plt.xlabel('x')
        plt.title('Método da Integração de 1/3 de Simpson para %s = %10.4f' %(funcao.replace('**','^').replace('*',''), val))
        
        plt.plot(x, y, '-', label = 'Função f(x)')
        plt.fill_between(x_b, y_b, 0 , facecolor = 'red', alpha = 0.5, label = "Área integrada")

        plt.legend(fancybox = True)
        plt.show()