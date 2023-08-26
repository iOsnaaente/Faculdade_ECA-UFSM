# Criado por Bruno Gabriel Flores Sampaio
# Universidade Federal de Santa Maria 
# Matricula 201720094
# Disciplina MTM 224 – Métodos Numéricos Computacionais – Turma 12 (em REDE)
# Professor Alex Andre Schimidt 

#QUESTÃO 3 :

#Utilize o Método de Newton para encontrar a raiz positiva da funcão

from __future__ import division
from numpy import *
import sympy  
import scipy.misc as sci
import matplotlib.pyplot as plt
x = sympy.symbols('x')

print('\n')

# Calcular a derivada da função
def newton(f,x0,TOL,N):
    i = 1
    # Calcu2lar f(x0) e df(x0)/dx
    fx0 = f(x0)

    # Calculo da derivada  
    dfx0 = sci.derivative(f,x0)
    # Calcula a primeira iteração do Método de Newton 
    z = x0 - (fx0/dfx0)
    
    # Calcula o valor de f no ponto z
    fx = f(z)

    # Primeiro critério de parada, calcular o intervalo
    intervalo = abs(z-x0)

    # Segundo critério de parada, se |f(x)| < TOL

    erro2 = abs(fx)

    while intervalo > TOL and erro2 > TOL  and i <= N:  
       
        x0 = z 
        fx0 = f(x0)
        dfx0 = sci.derivative(f,x0)

        z = x0 - (fx0/dfx0)

        fx = f(z)


        print("\n\n-----------------------------------------------------------------------------------------------------------------------------")
        print(" iteracao |  x(i)           | x(i+1)            |erro 1 = |x(i+1)-x(i)|  |  erro 2 = |f(x(i+1))|                          ") 
        print("-------------------------------------------------------------------------------------------------------------------------------")
        print("   %d      | x(%d)=%10.8f | x(%d)=%10.8f   | |x(%d)-x(%d)|=%10.8f | f(x(%d))=%10.8f  |  \n\n" % (i,i-1,x0,i,z,i,i-1,intervalo,i,abs(fx)))        
        
        #condicao de parada
        if (abs(fx) < TOL):

          print("O ponto x é solução\nA solução foi obtida através da imagem de f(x)\nO valor da solução é x = %8.6f e f(x) = %8.6f"%(z,fx))

          intervalo = linspace(z-20,z+20)
          plt.plot(z,f(z), 'go')
          plt.plot(intervalo,f(intervalo))
          plt.grid(); plt.show()

        i = i + 1

        intervalo = abs(z - x0)

    if (i >= N):  return print('Numero máximo de iterações excedido')
    if intervalo < TOL:

          print('O ponto x é solução')
          print('\n\nA solução foi obtida através do domínio de f(x)')
          print('\n\nO Valor da solução é:  x = %8.6f e f(x) = %8.6f'%(x,fx))

          intervalo = linspace(z-20,z+20)
          plt.plot(z,f(z), 'go')
          plt.plot(intervalo,f(intervalo))
          plt.grid()
          plt.show()


#Defina a funcao 
#func = input("Entre com uma função (ex: cos(x),x**2 ...): ")
func = '4*cos(x) - e**x'

f = lambda x: eval(func) 

# x0 - ponto inicial 
# TOL - Tolerancia
# N - Numero máximo de iterações 

#c = float(input("Entre com o ponto inicial: "))
c = 0.0 
#TOL = float(input("Entre com a tolerancia: "))
TOL = 0.000001
#N = int(input("Entre com o número máximo de iterações: ")) 
N = 100 

newton(f,c,TOL,N)


