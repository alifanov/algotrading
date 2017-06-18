import numpy as np
import backtrader as bt

from evostra import EvolutionStrategy
from keras.models import Model, Input, Sequential
from keras.layers import Dense, Activation


class ESStrategy(bt.Strategy):
    params = {
        'model': None
    }

    def __init__(self):
        self.order = None
        self.dataclose = self.datas[0].close
        self.datavol = self.datas[0].volume

    # def stop(self):
    #     cash = self.broker.getvalue()
    #     print('Result cash: {}'.format(cash))

    def next(self):
        if self.order:
            return

        inp = np.asanyarray([self.dataclose[0], self.datavol[0]])
        inp = np.expand_dims(inp, 0)

        predict = self.p.model.predict(inp)[0]
        predict = np.argmax(predict)

        if not self.position:
            if predict == 0:
                self.order = self.buy()
        else:
            if predict == 1:
                self.order = self.sell()

        if not self.position:
            if predict == 1:
                self.order = self.sell()
        else:
            if predict == 0:
                self.order = self.buy()

data = bt.feeds.GenericCSVData(
    dataname='eur_usd_1d.csv',
    separator=',',
    dtformat=('%Y%m%d'),
    tmformat=('%H%M00'),
    datetime=0,
    time=1,
    open=2,
    high=3,
    low=4,
    close=5,
    volume=6,
    openinterest=-1
)

model = Sequential()
model.add(Dense(128, input_dim=2, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(2, activation='relu'))

model.compile(optimizer='Adam', loss='mse')


def get_reward(weights):
    model.set_weights(weights)
    cerebro = bt.Cerebro()
    cerebro.addstrategy(ESStrategy, model=model)
    cerebro.adddata(data)
    cerebro.addsizer(bt.sizers.FixedSize, stake=50)

    cerebro.run()
    return cerebro.broker.getvalue()

es = EvolutionStrategy(model.get_weights(), get_reward, population_size=50, sigma=0.1, learning_rate=0.001)
es.run(1000, print_step=100)