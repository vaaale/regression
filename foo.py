from keras.engine import Input
from keras.layers import Dense, Flatten, Conv1D, LSTM
from keras.models import Model
import numpy as np
from dataset import load_data

SEQ_LEN = 10
NB_FEAT = 6

input = Input(shape=(SEQ_LEN, NB_FEAT))
x = Conv1D(4, 2)(input)
x = Flatten()(x)
x = Dense(5)(x)
y = Dense(1)(x)
model = Model(input=input, output=y)
model.compile(loss='mse', optimizer='adam')

# input = Input(shape=(SEQ_LEN, NB_FEAT))
# x = LSTM(4, return_sequences=False)(input)
# y = Dense(1)(x)
# model = Model(input=input, output=y)
# model.compile(loss='mse', optimizer='adam')


x_train, y_train, x_test, y_test = load_data('data/train.csv', SEQ_LEN)
nb_samples = len(x_train)
# nb_samples = 10000
print('{} samples with 10 features'.format(nb_samples))
print('Output (y) is the sum of the features')
# x_train = np.random.randint(10, size=(nb_samples, SEQ_LEN, NB_FEAT))
# x_train = np.random.uniform(-1., 1., size=(nb_samples, SEQ_LEN, NB_FEAT))
# y_train = np.sum(np.sum(x_train, axis=2), axis=1)

print('Reshapeing input to fit the model')
x_train = x_train.reshape(nb_samples, SEQ_LEN, NB_FEAT)

print('Fitting model')
model.fit(x_train, y_train, batch_size=32, nb_epoch=30)

# print('Testing model')
# x_test = np.random.uniform(-1., 1., size=(1, SEQ_LEN, NB_FEAT))
# correct_answer = np.sum(np.sum(x_test, axis=2), axis=1)
#
# output = model.predict(x_test)
# print('The model predicted {} and the correct answer should be {}'.format(output, correct_answer))




