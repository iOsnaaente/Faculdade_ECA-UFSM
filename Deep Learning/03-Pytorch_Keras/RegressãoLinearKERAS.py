import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense

input_dim = 1
output_dim = 1
lr = 0.0004

np.random.seed(2)

dados = pd.read_csv('dataset.csv', sep=',', skiprows=32, usecols=[2,3])
print(dados)

dados.plot.scatter(x='Age', y='Systolic blood pressure')
plt.xlabel('Idade (Anos)')
plt.ylabel('Pressão sistólica (mmHg)')
plt.show()

x = dados['Age'].values
y = dados['Systolic blood pressure'].values


modelo = Sequential()

modelo.add( Dense( output_dim, input_dim = input_dim, activation='linear') )

sgd = SGD(lr=lr)

modelo.compile(loss='mse', optimizer=sgd)
modelo.summary()


#TREINAMENTO 
num_epochs = 100000
batch_size = x.shape[0]
history = modelo.fit( x, y, epochs = num_epochs, batch_size=batch_size, verbose = 0 )

capas = modelo.layers[0]
w, b = capas.get_weights()
print('parametros: w = {:.1f}, b = {:.1f}'.format(w[0][0], b[0]))


plt.subplot(1,2,1)
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('ECM')
plt.title('ECM vs epochs')

y_regr = modelo.predict(x)

plt.subplot(1,2,2)
plt.scatter(x,y)
plt.plot(x,y_regr,'r')
plt.xlabel('x')
plt.ylabel('y')
plt.title('dados originais')
plt.show()


#Previsão 
x_prev = np.array([90])
y_prev = modelo.predict(x_prev)