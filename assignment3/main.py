import numpy as np
from dataloader import dataloader
from dependecy_parsing import get_configuration
from features import feature_vector, config_to_feature, transition_to_feature
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from gensim.models import KeyedVectors


def dependency_parser():

	model = KeyedVectors.load_word2vec_format('data/gensim_glove_vectors.txt')	

	sent_train, parse_train = dataloader('data/train.txt')
	sent_val, parse_val = dataloader('data/validation.txt')
	sent_test, parse_test = dataloader('data/test.txt')

	print('Loading {} training, {} validation and {} test sentences'.format(len(sent_train), len(sent_val), len(sent_test)))

	config_train, transit_train = get_configuration(sent_train, parse_train)
	config_val, transit_val = get_configuration(sent_val, parse_val)
	config_test, transit_test = get_configuration(sent_test, parse_test)

	print('Loading {} training, {} validation and {}  test configurations'.format(len(config_train), len(config_val), len(config_test)))

	x_train = config_to_feature(config_train,sent_train, parse_train)
	y_train = transition_to_feature(transit_train)
	print(x_train.shape, y_train.shape)

	x_val = config_to_feature(config_val, sent_val, parse_val)
	y_val = transition_to_feature(transit_val)
	print(x_val.shape, y_val.shape)

	x_test = config_to_feature(config_test, sent_test, parse_test)
	y_test = transition_to_feature(transit_test)
	print(x_test.shape, y_test.shape)

	model = Sequential()
	model.add(Dense(400, input_dim=707, activation='relu', use_bias=True))
	model.add(Dropout(0.3))
	model.add(Dense(100, activation='relu', use_bias=True))
	model.add(Dropout(0.3))
	model.add(Dense(3, activation='sigmoid'))
	model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])

	print(model.summary())

	model.fit(x_train, y_train, epochs=3, batch_size=32, validation_data=(x_val, y_val), verbose=1)

	score, accuracy = model.evaluate(x_test, y_test, batch_size=32, verbose=1)

	print ("accuracy: %.2f" % (accuracy))

if __name__ == "__main__":
	dependency_parser()