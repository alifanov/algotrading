import pandas as pd
import numpy as np

np.random.seed(7)
import math

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

from statsmodels.graphics.tsaplots import plot_acf

look_back = 1
features = 2

df = pd.read_csv('eur_usd_1d.csv', usecols=[5])
df_close = df['<CLOSE>']
dataset = df_close.values[:300]
dataset = dataset.astype('float32')

# plot autocorrelatio
# plot_acf(dataset)
# plt.show()

dataset = np.expand_dims(dataset, axis=1)

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


def create_model(hidden_neurons):
    model = Sequential()
    model.add(LSTM(hidden_neurons, input_shape=(look_back, trainX.shape[1])))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

model = KerasRegressor(build_fn=create_model, verbose=1, batch_size=1)
epochs = [10, 20] # , 50, 100, 200]
hidden_neurons = [4, 8] #, 16, 32, 64, 128]
grid_params = dict(epochs=epochs, hidden_neurons=hidden_neurons)
grid = GridSearchCV(estimator=model, param_grid=grid_params, n_jobs=-1)
grid_result = grid.fit(trainX, trainY)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))