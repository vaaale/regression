from tensorflow.contrib.keras.python.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.contrib.keras.python.keras.engine import Input
from tensorflow.contrib.keras.python.keras.engine import Model
from tensorflow.contrib.keras.python.keras.layers import LSTM, Dense
from tensorflow.contrib.keras.python.keras.optimizers import RMSprop

from dataset import load_data

SEQ_LEN = 10
NB_FEAT = 6

# input = Input(shape=(SEQ_LEN, NB_FEAT))
# x = Conv1D(4, 2)(input)
# x = Flatten()(x)
# x = Dense(5)(x)
# y = Dense(1)(x)
# model = Model(input=input, output=y)
# model.compile(loss='mse', optimizer='adam')

input = Input(shape=(SEQ_LEN, NB_FEAT))
x = LSTM(50, return_sequences=False, activation='tanh', name='LSTM1')(input)
y = Dense(1, activation='linear')(x)
model = Model(inputs=input, outputs=y)
model.compile(loss='mse', optimizer=RMSprop())
model.summary()


x_train, y_train, x_test, y_test = load_data('data/train.csv', SEQ_LEN)
nb_samples = len(x_train)
print('{} samples with {} features'.format(nb_samples, NB_FEAT))
print('Output (y) is the sum of the features')

print('Reshapeing input to fit the model')
x_train = x_train.reshape(nb_samples, SEQ_LEN, NB_FEAT)

print('Fitting model')
tb = TensorBoard(log_dir='logs/run-2', histogram_freq=0, write_graph=True, write_images=True, embeddings_freq=1)
checkpoint = ModelCheckpoint('logs/model-{epoch:02d}-{val_loss:.6f}.hdf5', monitor='val_loss',
                             save_best_only=True,
                             mode='min', save_weights_only=True, verbose=1)
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=128, epochs=300, callbacks=[tb, checkpoint])

