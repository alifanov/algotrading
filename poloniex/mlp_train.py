import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math
from keras.models import load_model
from sklearn.externals import joblib


np.random.seed(7)

look_back = 1
epochs = 1000
batch_size = 32
learning_rate = 1e-5


def get_model():
    model = Sequential()
    model.add(Dense(128, input_dim=look_back))
    # model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128))
    # model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128))
    # model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    return model


def get_loaded_model():
    return load_model('model.h5')


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


if __name__ == "__main__":
    df = pd.read_csv('btc_etc.csv')
    close_df = df['Close'].values.astype('float32')
    close_data = close_df.reshape(len(close_df), 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    close_data = scaler.fit_transform(close_data)

    joblib.dump(scaler, open('scaler.sav', 'wb'))

    train_size = int(len(close_data) * 0.67)
    test_size = len(close_data) - train_size
    train, test = close_data[0:train_size, :], close_data[train_size:len(close_data), :]

    print('Split data into training set and test set... Number of training samples/ test samples:', len(train),
          len(test))

    # convert stock price data into time series dataset
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    model = get_model()
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size)
    model.save('model.h5')
    # plt.plot(history.history['loss'])
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions and targets to unscaled
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.6f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.6f RMSE' % (testScore))

    draw = True
    if draw:
        # shift predictions of training data for plotting
        trainPredictPlot = np.empty_like(close_data)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

        # shift predictions of test data for plotting
        testPredictPlot = np.empty_like(close_data)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(close_data) - 1, :] = testPredict

        plt.plot(scaler.inverse_transform(close_data))
        plt.plot(trainPredictPlot)
        plt.plot(testPredictPlot)
        plt.show()
