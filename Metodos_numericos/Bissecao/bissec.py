from pandas import DataFrame
from numpy import linspace
from sympy import symbols
from math import *

# Primeiro definimos a variável que será lida f(x) = x 
x = symbols('x')


#Definição da função que encontra as raízes reais da função
def bissec(f, A, B, e):

    #Valor de A e B na função
    fA = f(A)
    fB = f(B)

    i = 0

    data = {
        'A'          : [],
        'B'          : [],
        'Pm'         : [],
        'f(A)'       : [],
        'f(B)'       : [],
        'f(Pm)'      : [],
        'Intervalo'  : [],
        'Tolerância' : []
    }

    if fA*fB > 0:
        return data

    else:
        #Definição do intervalo para parada posterior  
        intervalo = abs(B-A)

        #Ponto médio
        Pm = (A+B)/2
        
        while intervalo > e: 

            data['A'].append(A)   
            data['B'].append(B)
            data['Pm'].append(Pm)      
            data['f(A)'].append(f(A))    
            data['f(B)'].append(f(B))      
            data['f(Pm)'].append(f(Pm))   
            data['Intervalo'].append(intervalo)
            data['Tolerância'].append(e)

            if f(Pm) is 0:
                return data
            
            #Condições de zero em um intervalo 
            if f(A) * f(Pm) > 0: 
                A = Pm 
            else:
                B = Pm
            
            Pm = (B+A)/2
            
            i = i + 1
            intervalo = abs(B-A)

        return data 

# FUNÇÃO MAIN
if __name__ == '__main__':

    # INSTRUÇÕES
    print(' #########################################################')
    print(' ###             EXEMPLO DE ENTRADA:                   ###')
    print(' ###       x**2 + 5*x - 4  ou    cos(x)* x**2          ###') 
    print(' #########################################################\n')

    """
    # Entrada da função escolhida para o método
    funcao = input("Entre com uma funcao f(x): ")

    # Aplica a função no valor de x dado de entrada
    A = input("Entre com o limite inferior A: ")
    B = input("Entre com o limite superior B: ")
    e = input("Entre com o erro máximo permitido: ")
    
    """
    # Para testes
    funcao = "x**2  -5*x -30 "
    A = 0
    B =  10
    e = 0.0001
    
    
    # Para usar a função no ponto, aplicar f(p)
    f = lambda x: eval(funcao)

    dt = bissec(f, A, B, e)

    data = DataFrame(dt)
    print(data)

    str_funcao = funcao.replace('**','^').replace('*','').replace(' ','')

    try:
        print("\nA raiz de %s no intervalo (%10.8f, %10.8f) é: %10.8f \n" %(str_funcao, A,B, dt['Pm'][-1]))
        raiz = True
    except:
        print("Não existe raiz no intervalo [%10.8f,%10.8f] para a função f(x)=%s" %(A, B, str_funcao))
        raiz = False

    try:
        import matplotlib.pyplot as plt
        matplot = True

    except: 
        print("MatplotLib indisponível!!")
        matplot = False

    A = 1 if A<0 else A 
    B = 1 if B<0 else B 
    
    if raiz and matplot: 

        num2plot = linspace(A, B, (abs(A)+abs(B))*50 )
        x = []
        y = []
        
        for dot in num2plot:
            x.append(dot)
            y.append(f(dot))

        pontosX, pontosY = dt['Pm'], dt['f(Pm)']
        pontoX, pontoY   = dt['Pm'][-1], dt['f(Pm)'][-1]

        plt.ylabel('f(x)')
        plt.xlabel('x')
        plt.title('Método da Bisseção para %s' %str_funcao)
        
        plt.plot(x, y, '-', pontosX, pontosY, '+', pontoX, pontoY, 'o')

        plt.show()

    elif matplot:
        num2plot = linspace(A, B, (abs(A)+abs(B))*50 )
        x = []
        y = []
        
        for dot in num2plot:
            x.append(dot)
            y.append(f(dot))
        
        plt.ylabel('f(x)')
        plt.xlabel('x')
        plt.title('Método da Bisseção para %s' %str_funcao)
        
        plt.plot(x, y, '-')

        plt.show()



