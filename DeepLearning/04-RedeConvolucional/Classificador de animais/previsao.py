import numpy as np 
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

length, width = 100,100
modelo = './modelo/modelo.h5'
pesos = './modelo/pesos.h5'

cnn = tf.keras.models.load_model(modelo)
cnn.load_weights(pesos)

def previsao(file):
    x=load_img(file, target_size=(length,width))
    x=img_to_array(x)
    x=np.expand_dims(x, axis=0)
    arranjo = cnn.predict(x) ## [1,0,0]
    resultado = arranjo[0] 
    resposta = np.argmax(resultado)

    if resposta == 0:
        print("1 Catiorro")
    elif resposta == 1:
        print("És um Gatito")
    elif resposta == 2:
        print("Uno Gorilita")

    return resposta


#NOME DO ARQUIVO DENTRO DO PREVISAO( AQUI!!! )
previsao('ZUMA.jpg')
