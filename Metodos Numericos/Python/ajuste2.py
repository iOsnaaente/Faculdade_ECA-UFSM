import matplotlib.pylab as plt
import numpy as np
import math 
from sympy import *


#recebe a coluna[i] da matriz, e number=i atual

def pivo(col,number):
	p = -1
	for i,x in enumerate(col):
		if p == -1:
			if x != 0 and i>=number:
				p = i
		else:
			return p
	return p

def trocar_linhas(matriz,pivot,i):
	print(matriz)
	print(str(i)+"x"+str(pivot))
	aux = matriz[i,:].copy()
	matriz[i,:] = matriz[pivot,:]
	matriz[pivot,:] = aux
	print(matriz)
	return matriz


def subtrai(matriz, line1, line_pivot,m):
	matriz[line1,:] -= m*matriz[line_pivot,:] 
	return matriz


def escalonar(matriz,n):
	print(matriz)
	for i in range(n):
		col = matriz[:,i]
		pivot = pivo(col,i)
		#se o pivo n estiver na linha i, devem ser trocadas as linhas
		if pivot != i :
			matriz = trocar_linhas(matriz,pivot,i) 
			pivot = i


		#agora, zerar tds os valores da coluna, exceto o pivo, fazendo operacoes nas linhas 
		for x in range(n):
			#comeca a partir da linha abaixo do pivo
			if x > i:
				if col[x]!=0:
					m12 = col[x]/col[pivot]

					matriz = subtrai(matriz,x,pivot,m12)
	return matriz

def solucao(matriz):
	n = matriz.shape[0]
	vector = np.zeros(n, dtype = float)
	for i in reversed(range(n)):
		soma = 0
		for x in range(i, n):
			soma+= (vector[x]*matriz[i][x])

		vector[i] = (matriz[i][n] - soma)/matriz[i][i]

	return vector

def plotar_tudo(x,y,x1,y1):
	#plt.style.use('fivethirtyeight')
	plt.style.use('seaborn-darkgrid')

	plt.xlabel("Eixo x") 
	plt.ylabel("Eixo y") 
	
	plt.plot(x1, y1, 'green')
	
	#vertical e horizontal nos eixos
	plt.axhline(0, color='black')
	plt.axvline(0, color='black') 
	
	# Draw point based on above x, y axis values.
	plt.scatter(x, y, s=16, c='red')
	plt.show()


# x y axis value list.
x_number_list = []
y_number_list = []

x = Symbol('x')

#lendo os vetores digitados pelo usuario
X = np.array([i for i in list(map(float,input('vetor x = ').split()))])
Y = np.array([i for i in list(map(float,input('vetor y = ').split()))])
n = len(X)


plotar_tudo(X,Y, 0, 0)

F = ''

resp = input("Ajuste é polinomial? (s/n)")

if resp == "s":
	gr = int(input("Grau do polinomio: "))


		#passos pra resolver o sistema

	#construindo matriz A
	A = np.array([[sum(x**(2*gr-i-j) for x in X) for i in range(gr+1)] for j in range(gr+1)])
	
	b =  np.array([[sum(y*x**(gr-i)for x,y in zip(X,Y))] for i in range(gr+1)])

	#juntando matriz A e b
	Ab = np.hstack((A,b))
	print(Ab)

	solucao = solucao(escalonar(Ab,gr+1))

	#funcao aproximada
	for i,x in enumerate(solucao):
		F+="+"+str(x)+"*x**"+str(gr-i)
	print(F)
	# F = eval("lambda x:" + F)

	# x1 = np.arange(X[0],X[-1],0.1)
	# y1 = F(x1)
	# plotar_tudo(X,Y, x1, y1)

else:

	ajuste = input("Qual é o tipo de ajuste? (a/b/c)")

	X1 = X.copy()
	Y1 = Y.copy()

	if ajuste == 'a':
		print("f(x) = 1/ax+b")
		Y1 = 1/Y
	if ajuste == 'b':
		print("f(x) = alfa*e^(Beta*x)")
		Y1 = np.log(Y)
	if ajuste == 'c':
		print("f(x) = alfa*x^Beta")
		Y1 = np.log(Y)
		X1 = np.log(X)

	gr = 1 
		#passos pra resolver o sistema

	#construindo matriz A
	A = np.array([[sum(x**(2*gr-i-j) for x in X1) for i in range(gr+1)] for j in range(gr+1)])
	
	b =  np.array([[sum(y*x**(gr-i)for x,y in zip(X1,Y1))] for i in range(gr+1)])

	#juntando matriz A e b
	Ab = np.hstack((A,b))
	print(Ab)

	solucao = solucao(escalonar(Ab,gr+1))
	print(solucao)

	if ajuste == "a":
		F = '1/('+str(solucao[0])+'*x+'+str(solucao[1])+')'
	if ajuste == "b":
		solucao[1] = np.exp(solucao[1])
		F = str(solucao[1])+"*np.e**("+str(solucao[0])+"*x)"


	if ajuste == "c":
		solucao[1] = np.exp(solucao[1])
		F = str(solucao[1])+"*x**"+str(solucao[0])

#funcao aproximada

print(F)
F = eval("lambda x:" + F)

erro = 0
for x,y in zip(X,Y):
	erro += (y-F(x))**2
print('ET^2 = ',erro)

x1 = np.arange(X[0],X[-1]+0.1,0.1)
y1 = F(x1)

plotar_tudo(X,Y, x1, y1)





