import matplotlib.pyplot as plt
from numpy import linspace
from math import*


# Função da secante 
def interpolation(X,Y):

    Xn = linspace(X[0], X[-1], 100)
    Yn = []

    for xn in Xn:

        val = 0 
        for i in range(len(X)):
            L = 1
            for j in range(len(X)):
                if i is not j :
                    L = L*((xn-X[j])/(X[i]-X[j]))

            val = val + Y[i]*L

        Yn.append(val)

    return [Xn,Yn]


if __name__ == "__main__":    

    # INSTRUÇÕES
    #x  = map(float, input("Entre com o vetor de X: ").split(' '))
    #y  = map(float, input("Entre com o vetor de Y: ").slipt(' '))

    x = [0,1,2,3,4,5,6,7,8,9]
    y = [0,1,2,3,4,2,7,2,8,3]

    xn, yn = interpolation(x, y)
    
    plt.ylabel('f(x)')
    plt.xlabel('x')
    plt.title('Método da Interpolação')
    
    plt.plot(x , y , 'o', label='Pontos definidos')
    plt.plot(xn, yn, '-', label = 'Função Interpolada')
    
    plt.legend(fancybox = True)

    plt.show()
