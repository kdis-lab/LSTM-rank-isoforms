from sklearn.model_selection import StratifiedKFold
import numpy
import keras
import json
import os
from os.path import join
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight
import sys
import time
from multiprocessing import Pool
from keras.preprocessing import sequence

from rna_pandas_script_without0 import get_data
from callback_measures import SingleLabel, Binary, AccumulativeTime, EpochsRegister, SingleLabelMultioutput, BinaryMultioutput

def one_fold(mapitems):
	# ocupa la memoria gpu necesaria
	from keras import backend as k
	import tensorflow as tf
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	k.set_session(sess)
	
	epochs = mapitems['epochs']
	batch = mapitems['batch']
	model = mapitems['model']
	X = mapitems['X']
	Y = mapitems['Y']
	Xtest = mapitems['Xtest']
	Ytest = mapitems['Ytest']
	optimizer = mapitems['optimizer']
	dirpath = mapitems['dirpath']
	metric = mapitems['metric']
	metric_mode = mapitems['metric_mode']

	train_model = model['model'](X, model['units'])

	train_model.compile(optimizer=optimizer, loss='binary_crossentropy',
						metrics=['accuracy'])

	# callbacks
	callbacks = [AccumulativeTime(),BinaryMultioutput(Xtest, [Ytest,Ytest,Ytest]),SingleLabelMultioutput(Xtest, [Ytest,Ytest,Ytest])]
	
	if optimizer == "sgd":
		callbacks.append(
			ReduceLROnPlateau(monitor=metric, factor=0.6, mode=metric_mode, verbose=1))
	callbacks.append(EpochsRegister(join(dirpath, 'epochs.txt'),
									join(dirpath, 'epochs-mean.txt')))
	# end callbacks

	train_model.fit(X, [Y,Y,Y], epochs=epochs, batch_size=batch,
						validation_data=(Xtest, [Ytest,Ytest,Ytest]),
						callbacks=callbacks,
						verbose=2)		


def kfold(config_file, models):
	# code
	with open(config_file) as json_data:
		configuration = json.load(json_data)

	folds = int(configuration['folds'])
	epochs = int(configuration['epochs'])
	seed = int(configuration['seed'])
	reportsDir = configuration['reportsDir']
	metric = configuration['metric']
	metric_mode = configuration['metric_mode']

	for dataset in configuration['datasets']:
		for gen in dataset['genes']:
			x1,x2,y = get_data(dataset['file'], gen)
			# top
			max_review_length = max([len(i) for i in x1])
			x1 = sequence.pad_sequences(x1, maxlen=max_review_length)
			x1 = numpy.reshape(x1, (x1.shape[0], x1.shape[1], 1))
			# low
			max_review_length = max([len(i) for i in x2])
			x2 = sequence.pad_sequences(x2, maxlen=max_review_length)
			x2 = numpy.reshape(x2, (x2.shape[0], x2.shape[1], 1))
			xs = [				
				{'n': 'two', 'x': [x1,x2]}
			]
			for ix in xs:
				for batch in dataset['batch']:
					for model in models:
						for optimizer in configuration['optimizers']:
							num_batch = int(batch)

							dirpath = join(reportsDir, dataset['name'], "gen_" + str(gen),
											model['name'] + '_' + ix['n'], "batch_" + str(batch),
											optimizer)

							try:
								# if this experiment was finished continue
								if os.path.exists(join(dirpath, 'summary.txt')):
									continue
								else:
									# if not, delete the partial results
									if os.path.exists(join(dirpath, 'epochs-mean.txt')):
										os.remove(join(dirpath, 'epochs-mean.txt'))
									if os.path.exists(join(dirpath, 'epochs.txt')):
										os.remove(join(dirpath, 'epochs.txt'))

								if not os.path.exists(dirpath):
									os.makedirs(dirpath)

								# fix random seed for reproducibility
								numpy.random.seed(seed)

								# define kfold cross validation test harness
								countsclasses = numpy.bincount(y)
								minclass = numpy.argmin(countsclasses)
								minclasscounts = countsclasses[minclass]
								
								kfold = StratifiedKFold(n_splits=minclasscounts, shuffle=True, random_state=seed)
								
								fold = 0
								xtemp = ix['x']
								for train, test in kfold.split(xtemp[0], y):
									
									if len(xtemp) == 2:
										xtrain = [xtemp[0][train],xtemp[1][train]]
										xtest = [xtemp[0][test],xtemp[1][test]]
									else:
										xtrain = xtemp[train]
										xtest = xtemp[test]
					
									time.sleep(5)
									start_time = time.time()
									with Pool(processes=1) as pool:
										pool.map(one_fold, [{
											'fold': fold,
											'epochs': epochs,
											'batch': num_batch,
											'model': model,
											'X': xtrain,
											'Y': y[train],
											'Xtest': xtest,
											'Ytest': y[test],
											'optimizer': optimizer,
											'dirpath': dirpath,
											'metric': metric,
											'metric_mode': metric_mode
										}])
									print('>> fold', dirpath)
									print('>> fold', fold, 'completed in', str(time.time() - start_time), 'seconds')
									fold = fold + 1

								final_evaluations = numpy.genfromtxt(join(dirpath, 'epochs-mean.txt'), delimiter=',',
																	dtype=numpy.float64, names=True)
								# evaluaciones de una metrica
								metric_column = final_evaluations[metric]
								# el indice de la fila donde esta la mejor metrica
								row = metric_column.argmax() if metric_mode == 'max' else metric_column.argmin()

								evaluation = final_evaluations[row]
								summary = open(join(dirpath, 'summary.txt'), mode='w')
								# leer las keys, las metricas
								summary.write(','.join(final_evaluations.dtype.names))
								summary.write('\n')
								summary.write(','.join(map(str, evaluation)))
								summary.close()
								
								time.sleep(5)
							except Exception as exception:
								print('error >>', dirpath)
								print('reason >>', exception)


if __name__ == '__main__':
	config_file = str(sys.argv[1])
	from lstm_model_parallel_timesteps import lstm_all
	# from utils_keras.lstm.lstm_model_conv import lstm_100_conv_maxpooling

	kfold(config_file, [
		lstm_all(10)
	])
