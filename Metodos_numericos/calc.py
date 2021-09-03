# Criado por Bruno Gabriel Flores Sampaio em 30/12/2020
# Universidade Federal de Santa Maria 
# Matricula 201720094
# Disciplina MTM 224 – Métodos Numéricos Computacionais – Turma 12 (em REDE)
# Professor Alex Andre Schimidt 

#QUESTÃO 2 :

#Aplique o MÉtodo de Gauss-Siedel com 4 casas decimais de precisÃo e tolerância e<10^-3

x1 = lambda d,t   : (-6 -d +2*t ) /-10
x2 = lambda u,t,q : (25 + u + t -4*q) /11
x3 = lambda u,d,q : (-11 -2*u + d + q) / 10
x4 = lambda d,t   : (15 -3*d + t) /8

from math import inf 

vet = [0,0,0,0]

val = [0, 0, 0, 0]
erro = [inf, inf, inf, inf ]

erroMax = 10**(-3)

i = 0 

prints = []

while max(erro) > erroMax :
    pass
    val[0] = x1( vet[1], vet[2] )
    print("X1: (-6 - %2.4f + 2.%2.4f )/-10" %(vet[1], vet[2] ))
    val[1] = x2( val[0], vet[2], vet[3] )
    print("X2: (25 + %2.4f + %2.4f -4.%2.4f)/11" %( val[0], vet[2], vet[3] )  )
    val[2] = x3( val[0], val[1], vet[3] )
    print("X3: (-11 - 2.%2.4f + %2.4f + %2.4f)/10" %( val[0], val[1], vet[3]  )  )
    val[3] = x4( val[1], val[2])
    print("X4: (15 - 3.%2.4f + %2.4f)/8" %( val[1], val[2])  )

    erro = [ abs(vet[i]-val[i]) for i in range(len(vet)) ]

    vet = [ val[i] for i in range(len(val)) ]
    i = i + 1

    print("ident %i \tval1 %2.4f\tval2 %2.4f\tval3 %2.4f\tval4 %2.4f \tErroMax = %2.4f \n\n" % ( i, val[0], val[1], val[2], val[3], max(erro) ) )