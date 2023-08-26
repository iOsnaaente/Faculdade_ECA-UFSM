from pandas import DataFrame
from numpy import linspace
from sympy import symbols
from math import*

# Primeiro definimos a variável que será lida f(x) = x 
x = symbols('x')

# Função da secante 
def secante(f, x0, x1, TOL, N):

    # Número de iterações
    i = 1

    # Calcular f(x0) e f'(x0)
    fx0 = f(x0)
    fx1 = f(x1) 

    # Calcula a primeira iteração do Método de Newton 
    z_2 = x1 - fx1*(x1 - x0) / ((fx1 - fx0) + 0.000001)

    # Calcula o valor de f no ponto z
    fx = f(z_2)

    # Primeiro critério de parada, calcular o intervalo
    intervalo = abs(z_2 - x0)

    z_1 = x1

    # DataFrame para salvar os dados impressos na tela com o pandas
    data = {'x(i)': [],'x(i+1)':[],'f(x)':[], '|x(i+1)-x(i)|':[],'|f(x(i+1))|':[], 'Tolerancia':[] }

    while True:  

        fx0 = f(x0)

        z = z_2 - f(z_2)*(z_2 - z_1)/(f(z_2) - f(z_1)+0.0000001)
        
        z_1, z_2 = z_2, z

        fx = f(z)

        data['x(i)'].append(x0)
        data['x(i+1)'].append(z)
        data['f(x)'].append(f(x0+i))
        data['|x(i+1)-x(i)|'].append(abs(z-x0))
        data['|f(x(i+1))|'].append(abs(fx))
        data['Tolerancia'].append(TOL)
        
        i = i + 1

        intervalo = abs(z - x0)

        # Condição de parada iteração máxima excedida 
        if (i >= N):
            dt = DataFrame(data)
            print(dt, end='\n')
            print('\nNúmero máximo de iterações excedido!\n')
            print("Limite não possui singularidade!!!\n")
            break           

        #Condição de parada - 
        if abs(fx) < TOL:
            dt = DataFrame(data)
            print(dt, end='\n')
            print('\nO ponto x é solução')
            print('A solução foi obtida através do imagem de f(x)')
            print('O Valor da solução é:  x=%8.6f e f(x)=%8.6f\n'%(z,fx))
            break

        # Condição de parada - Ponto encontrado
        if intervalo < TOL:
            dt = DataFrame(data)
            print(dt, end='\n')
            print('\nO ponto x é solução')
            print('A solução foi obtida através do domínio de f(x)')
            print('O Valor da solução é:  x=%8.6f e f(x)=%8.6f \n'%(x,fx))
            break 

    return data

if __name__ == "__main__":    

    # INSTRUÇÕES
    print(' #########################################################')
    print(' ###             EXEMPLO DE ENTRADA:                   ###')
    print(' ###       x**2 + 5*x - 4  ou    cos(x)* x**2          ###') 
    print(' #########################################################\n')

    '''
    # Entrada da função escolhida para o método
    funcao = input("Entre com uma funcao f(x): ")

    x0  = float(input("Entre com o ponto inicial (x0): "))
    x1  = float(input("Entre com o ponto inicial (x1): "))
    TOL = float(input("Entre com a tolerancia: "))
    N   = int(input("Entre com o nuhmero mahximo de iteracoes: ")) 
    
    '''
    # Para testes
    funcao = "x**3  -5*x +3 "
    x0 = -10
    x1 = 5
    TOL = 0.0001
    N = 10000
    
    # Aplica a função no valor de x dado de entrada
    f = lambda x: eval(funcao) 
    
    dt = secante(f, x0, x1, TOL, N)

    try:
        import matplotlib.pyplot as plt
        matplot = True

    except: 
        print("MatplotLib indisponível!!")
        matplot = False
        
    if matplot:
        num2plot = linspace(x0, x1, (abs(x0)+abs(x1))*100 )
        x = []
        y = []
        for dot in num2plot:
            x.append(dot)
            y.append(f(dot))

        pontoX, pontoY = dt['x(i+1)'], dt['|f(x(i+1))|']
        X, Y = dt['x(i+1)'][-1], dt['|f(x(i+1))|'][-1]

        plt.ylabel('f(x)')
        plt.xlabel('x')
        plt.title('Método da Secante para %s' %funcao.replace('**','^').replace('*',''))
        
        plt.plot(x, y, '-', pontoX, pontoY, '+', X, Y, 'o')

        plt.show()
