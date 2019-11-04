import keras
from keras.models import Model
from keras.layers import Input, MaxPooling1D, LSTM, Embedding, Dense
import numpy

	
def model_all(x=None, units=1):
	import tensorflow as tf
	with tf.device('/gpu:0'):
		in1 = Input(shape=(x[0].shape[1], 1))
		v1 = LSTM(units)(in1)
		v1 = Dense(1, activation='sigmoid')(v1)
	
	with tf.device('/gpu:1'):
		in2 = Input(shape=(x[1].shape[1], 1))
		v2 = LSTM(units)(in2)
		v2 = Dense(1, activation='sigmoid')(v2)

	out = keras.layers.average([v1, v2], name="twomix")
	
	model = Model(inputs=[in1, in2], outputs=[v1,v2,out])
		
	return model

def lstm_all(units=1):
	return {"model": model_all, "units": units,"name": "lstmmixto_" + str(units)}