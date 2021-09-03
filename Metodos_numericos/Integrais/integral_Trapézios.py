from numpy import linspace
from sympy import symbols 

from math import *

# Definição da variável em uso -> f(x): x
x = symbols('x')

def Trapezio(f, a, b, e):
        
    #Distancia entre os intervalos
    d = 0.001

    xd = []
    yd = []
    for i in linspace(a,b,round((b-a)/d)):
        # Vetor de m pontos cobrindo o intervalo [A,B]
        xd.append(i)
        # Vetor que pega a derivada de xd em m[i]
        yd.append( (f(i+2*d) - 2*f(i+d) + f(i) )/(d**2) )

    #Valor máximo da derivada segunda dos subintervalos 
    Max = max(yd)
    Min = min(yd)
    M = abs(Max) if abs(Max)>abs(Min) else abs(Min) 

    m = sqrt((((b-a)**3)*M)/(12*e))

    ##Arredondar o numero para cima
    if m > round(m):
        m = round(m)+1
    else:
        m = round(m)
    
    #CALCULO DA INTEGRAL 
    h = (b-a)/m
    s = f(a) + f(b)
    x = a

    for i in range(m-1):
        x = x + h
        s = s + 2*f(x)

    #Valor aproximado da integral que sera devolvido
    Itr = (h/2)*s

    print("O valor da integral pelo método dos trapézios repetidos é: %8.10f \n" %Itr)
    return Itr

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
    e = 0.0001

    f = lambda x: eval(funcao) 

    val = Trapezio(f, A, B, e)

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
        plt.title('Método da Integração dos Trapézios repetidos para %s = %10.4f' %(funcao.replace('**','^').replace('*',''), val))
        
        plt.plot(x, y, '-', label = 'Função f(x)')
        plt.fill_between(x_b, y_b, 0 , facecolor = 'red', alpha = 0.5, label = "Área integrada")

        plt.legend(fancybox = True)
        plt.show()

