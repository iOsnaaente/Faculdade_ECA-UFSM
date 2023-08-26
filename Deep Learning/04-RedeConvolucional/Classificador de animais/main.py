from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import backend as K 
from tensorflow.python.keras import optimizers
import sys 
import os

K.clear_session()

data_treinamento = './data/treinamento'
data_validacao = './data/validacao'

#parametros

epocas = 20

length, width = 100, 100
batch_size = 32

passos = 100
passos_validacao = 32

filtrosConv1 = 32
filtrosConv2 = 64

tamanho_filtro1 = (3,3)
tamanho_filtro2 = (2,2)
tamanho_pool = (2,2)
classes = 3

lr = 0.0005


treinamento_datagen = ImageDataGenerator(
		rescale=1./255,
		shear_range = 0.3,
		zoom_range = 0.3,
		horizontal_flip = True
	)

validacao_datagen = ImageDataGenerator(
		rescale = 1./255
	)

imagem_treinamento = treinamento_datagen.flow_from_directory(
	data_treinamento,
	target_size=(length,width),
	batch_size=batch_size,
	class_mode='categorical'
)

imagem_validada = validacao_datagen.flow_from_directory(
	data_validacao,
	target_size=(length, width),
	batch_size= batch_size,
	class_mode= 'categorical'
)

cnn = Sequential()

cnn.add(
	Convolution2D(
		filtrosConv1,
		tamanho_filtro1, 
		padding='same', 
		input_shape=(length, width,3), 
		activation='relu'
		)
	)
cnn.add( MaxPooling2D( pool_size = tamanho_pool))


cnn.add(
	Convolution2D(
		filtrosConv2,
		tamanho_filtro2,
		padding= "same",
		activation='relu'
	)
)
cnn.add( MaxPooling2D( pool_size=tamanho_pool))

cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(classes, activation='softmax'))

cnn.compile(
	loss='categorical_crossentropy', 
	optimizer=optimizers.Adam(lr=lr), 
	metrics=['accuracy']
	)

cnn.fit_generator(
	imagem_treinamento, 
	steps_per_epoch = passos, 
	epochs = epocas, 
	validation_data = imagem_validada,
	validation_steps = passos_validacao
	)

target_dir = './modelo/'
if not os.path.exists(target_dir):
	os.mkdir(target_dir)
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')

