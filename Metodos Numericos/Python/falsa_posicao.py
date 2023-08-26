import numpy as np
import matplotlib.pyplot as plt

#A função já está definida aqui 
def f(x): return x**2-2

# a - extremo esquerdo 
# b - extremo direito 
# TOL - Tolerancia
# N - Numero máximo de iterações 

def falsaposicao(f,a,b,TOL,N):
    i = 1
    fa = f(a)
    fb = f(b)
    intervalo = abs(b-a)
    while intervalo > TOL and i <= N :  
        x = float((a*f(b) - b*f(a)) / (f(b)-f(a)))
        fa = f(a)
        fx = f(x)
        fb = f(b)
        #condicao de parada
        if abs(fx) < TOL:
          return print("O ponto x é solução\nA solução foi obtida através da imagem de f(x)\nO valor da solução é x = %8.6f e f(x) = %8.6f"%(x,fx))
        print('3')

        print('\n\n------------------------------------------------------------------------------------------------------------------------------')
        print(' iteracao |     a    |     f(a)     |     b     |   f(b)    |   x   |  f(x)  | erro 1(b-a) | erro 2(f(x))                          ')
        print('-------------------------------------------------------------------------------------------------------------------------------\n')
        print('   %d      | %f | %8.6f  |  %8.6f  | %8.6f  | %8.6f | %8.6f | %8.6f | %8.6f  \n\n '% (i,a,fa,b,fb,x,f(x),(b-a),abs(f(x))))

        i = i + 1
        #bissecta o intervalo 
        if (fa * fx > 0):
            a = x
            fa = fx
        else:
            b = x
        intervalo = abs(b-a)

    if (i >= N):  return print('Numero máximo de iterações excedido')
    if intervalo < TOL:
        print('O ponto x é solução')
        print('\n\nA solução foi obtida através do domínio de f(x)')
        print('\n\nO Valor da solução é:  x = %8.6f e f(x) = %8.6f'%(x,fx))
        
a = int(input("Entre com o ponto a: "))
b = int(input("Entre com o ponto b: "))
TOL = float(input("Entre com a tolerancia: "))
N = int(input("Entre com o número máximo de iterações: ")) 

falsaposicao(f,a,b,TOL,N)

intervalo = np.linspace(a,b)
plt.plot(intervalo,f(intervalo))
plt.grid(); plt.show()

