import time
import numpy as np

from es import EvolutionStrategy
from keras.models import Model, Input, Sequential
from keras.layers import Dense, Activation

import pandas as pd
import matplotlib.pyplot as plt
from stockstats import StockDataFrame


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r %2.4f sec' % (method.__name__, te - ts))
        return result

    return timed


class BaseIndicator:
    def __init__(self, title='', draw_inline=True):
        self.title = title
        self.draw_inline = draw_inline

    def draw_extra_charts(self, *args, **kwargs):
        pass


class SMAIndicator(BaseIndicator):
    def __init__(self, period, title='sma'):
        super(SMAIndicator, self).__init__(title)
        self.period = period

    def get_data(self, df, field_name):
        ds = df[field_name]
        return ds.rolling(center=False, window=self.period).mean()


class MomentumIndicator(BaseIndicator):
    def __init__(self, period=14, title='momentum'):
        super(MomentumIndicator, self).__init__(title)
        self.period = period

    def get_data(self, df, field_name):
        ds = df[field_name]
        if len(ds) > self.period - 1:
            return ds[-1] * 100 / ds[-self.period]
        return None


class RSIIndicator(BaseIndicator):
    def __init__(self, period=14, title='rsi'):
        super(RSIIndicator, self).__init__(title, draw_inline=False)
        self.period = period

    def draw_extra_charts(self, axe):
        axe.axhline(y=20, xmin=0, xmax=1, c='red', zorder=0, linewidth=1)
        axe.axhline(y=80, xmin=0, xmax=1, c='green', zorder=0, linewidth=1)

    def get_data(self, df, field_name):
        ds = df[field_name]
        delta = ds.diff()
        d_up, d_down = delta.copy(), delta.copy()
        d_up[d_up < 0] = 0
        d_down[d_down > 0] = 0

        rol_up = d_up.rolling(center=False, window=self.period).mean()
        rol_down = d_down.rolling(center=False, window=self.period).mean().abs()

        rs = rol_up / rol_down
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi


class EMAIndicator(BaseIndicator):
    def __init__(self, period, title='ema'):
        super(EMAIndicator, self).__init__(title)
        self.period = period

    def get_data(self, df, field_name):
        return df['{}_{}_ema'.format(field_name, self.period)]


class DataPoint:
    def __init__(self, dt):
        self.dt = dt

    def __getattr__(self, key):
        return self.dt[key]


class DataSeries:
    def __init__(self, data):
        self.data = StockDataFrame.retype(data.copy())
        self.index = 0
        self.indicators = []
        self.data.set_index('index')
        data_dict = self.data.to_dict(orient='split')
        self._data = data_dict['data']
        self._columns = data_dict['columns']

    def add_indicator(self, indicator):
        self.data[indicator.title] = indicator.get_data(self.data, 'close')
        self.indicators.append(indicator)
        data_dict = self.data.to_dict(orient='split')
        self._data = data_dict['data']
        self._columns = data_dict['columns']

    def get_dot(self, index):
        return {k: v for k,v in zip(self._columns, self._data[index])}

    def __getitem__(self, index):
        return DataPoint(self.get_dot(index + self.index))

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        value = self.get_dot(self.index)
        self.index += 1
        if self.index >= self.data.shape[0]:
            raise StopIteration
        return DataPoint(value)


class Trade:
    status = 'OPEN'
    open_price = None
    close_price = None
    volume = None
    open_datetime = None
    close_datetime = None

    def open(self, datetime, price, volume):
        self.status = 'OPEN'
        self.open_price = price
        self.volume = volume
        self.open_datetime = datetime

    def close(self, datetime, price):
        self.status = 'CLOSE'
        self.close_price = price
        self.close_datetime = datetime

    def get_profit(self):
        return self.volume*(self.close_price - self.open_price)


class BT:
    def __init__(self, ds, balance=1000.0):
        self.ds = ds
        self.balance = balance
        self.start_balance = balance
        self.end_balance = balance
        self.position = None
        self.trades = []

    def buy(self, price, shares_volume):
        trade = Trade()
        trade.open(self.ds.index, price, shares_volume)
        self.trades.append(trade)
        self.position = trade

    def sell(self, price):
        self.position.close(self.ds.index, price)
        self.balance += self.position.get_profit()
        self.position = None

    def process_bar(self):
        pass
        # if not self.position:
        #     current_diff = self.ds[0].close - self.ds[-1].close
        #     prev_diff = self.ds[-1].close - self.ds[-2].close
        #     prev_prev_diff = self.ds[-2].close - self.ds[-3].close
        #     if prev_prev_diff < 0 and prev_diff > 0 and current_diff > 0 and self.ds[0].volume > 5.0:
        #         # print('BUY: price {} with {} shares on volume {}'.format(self.ds[0].close, 1000.0, self.ds[0].volume))
        #         self.buy(self.ds[0].close, 1000.0)
        # else:
        #     if self.ds[0].close - self.ds[-1].close < 0:
        #         self.sell(self.ds[0].close)

    def stop(self):
        self.end_balance = self.balance

    def get_profit(self):
        return self.end_balance - self.start_balance

    def get_roi(self):
        return 1.0*(self.end_balance - self.start_balance)/self.start_balance

    def run(self):
        for _ in self.ds:
            self.process_bar()
        self.stop()

    def plot(self):
        data = self.ds.data
        data = data.set_index('datetime')
        headers = ['close']
        outline_indicators = []
        outline_data = {}
        for ind in self.ds.indicators:
            if ind.draw_inline:
                headers.append(ind.title)
            else:
                outline_indicators.append(ind)
                outline_data[ind.title] = data[ind.title]
                data.drop(ind.title, inplace=True, axis=1)

        data = data[headers]
        fig, axes = plt.subplots(nrows=len(outline_indicators) + 1, ncols=1)
        data.plot(ax=axes[0], sharex=True)
        for i, oi in enumerate(outline_indicators):
            outline_data[oi.title].plot(ax=axes[i+1], sharex=True)
            oi.draw_extra_charts(axes[i+1])

        # data.plot(marker='o', markersize=4)

        trade_open_datetimes = [trade.open_datetime for trade in self.trades]
        trade_open_prices = [trade.open_price for trade in self.trades]
        plt.plot(trade_open_datetimes, trade_open_prices, 'g^')

        trade_close_datetimes = [trade.close_datetime for trade in self.trades]
        trade_close_prices = [trade.close_price for trade in self.trades]
        plt.plot(trade_close_datetimes, trade_close_prices, 'rv')

        plt.show()


class SMABT(BT):
    def __init__(self, ds, balance, period):
        super(SMABT, self).__init__(ds, balance)
        self.ds.add_indicator(SMAIndicator(period=period))

    def process_bar(self):
        if not self.position:
            if self.ds[0].sma and self.ds[-1].close < self.ds[-1].sma:
                if self.ds[0].close > self.ds[0].sma:
                    self.buy(self.ds[0].close, 1000.0)
        else:
            if self.ds[0].sma and self.ds[-1].close > self.ds[-1].sma:
                if self.ds[0].close < self.ds[0].sma:
                    self.sell(self.ds[0].close)


class RSIBT(BT):
    def __init__(self, ds, balance, period):
        super(RSIBT, self).__init__(ds, balance)
        self.ds.add_indicator(RSIIndicator(period=period))

    def process_bar(self):
        if not self.position:
            if self.ds[0].rsi:
                if self.ds[0].rsi > 20.0 and self.ds[-1].rsi < 20.0:
                    self.buy(self.ds[0].close, 1000.0)
        else:
            if self.ds[0].rsi < 80.0 and self.ds[-1].rsi > 80.0:
                self.sell(self.ds[0].close)


class EMABT(BT):
    def __init__(self, ds, balance, period):
        super(EMABT, self).__init__(ds, balance)
        self.ds.add_indicator(EMAIndicator(period=period))

    def process_bar(self):
        if not self.position:
            if self.ds[0].ema and self.ds[-1].close < self.ds[-1].ema:
                if self.ds[0].close > self.ds[0].ema:
                    self.buy(self.ds[0].close, 1000.0)
        else:
            if self.ds[-1].close < self.ds[0].close:
                self.sell(self.ds[0].close)


def simple_sma():
    df = pd.read_csv('btc_etc.csv').rename(columns={
        'Close': 'close',
        'Date time': 'datetime',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Volume': 'volume'
    })
    ds = DataSeries(df)
    bt = SMABT(ds, balance=1000.0, period=200)
    bt.run()
    print('Profit: ${:.2f}'.format(bt.get_profit()))
    # bt.plot()


def simple_rsi():
    df = pd.read_csv('btc_etc.csv').rename(columns={
        'Close': 'close',
        'Date time': 'datetime',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Volume': 'volume'
    })
    ds = DataSeries(df)
    bt = RSIBT(ds, balance=1000.0, period=200)
    bt.run()
    print('Profit: ${:.2f}'.format(bt.get_profit()))
    # bt.plot()


def optimize_params():
    df = pd.read_csv('btc_etc.csv').rename(columns={
        'Close': 'close',
        'Date time': 'datetime',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Volume': 'volume'
    })

    best_period = 0
    best_profit = 0
    for n in range(3, 200):
        ds = DataSeries(df)
        bt = RSIBT(ds, balance=1000.0, period=n)
        bt.run()
        if best_profit < bt.get_profit():
            best_profit = bt.get_profit()
            best_period = n

        # print('Period: {}'.format(n))
        # print('Profit:\t\t${:.2f} ({:+.2%})'.format(bt.get_profit(), bt.get_roi()))
        # print('---')
    print('RSI best period')
    print('Period: {}'.format(best_period))
    print('Profit: ${:.2f}'.format(best_profit))

    best_period = 0
    best_profit = 0
    for period in range(2, 200):
        ds = DataSeries(df)
        bt = SMABT(ds, balance=1000.0, period=n)
        bt.run()
        if best_profit < bt.get_profit():
            best_profit = bt.get_profit()
            best_period = n

        # print('Period: {}'.format(n))
        # print('Profit:\t\t${:.2f} ({:+.2%})'.format(bt.get_profit(), bt.get_roi()))
        # print('---')
    print('SMA best period')
    print('Period: {}'.format(best_period))
    print('Profit: ${:.2f}'.format(best_profit))


def get_model():
    model = Sequential()
    model.add(Dense(16, input_dim=12, activation='relu'))
    # model.add(Dense(256, activation='relu'))
    model.add(Dense(3, activation='relu'))

    model.compile(optimizer='Adam', loss='mse')
    return model


class NNBT(BT):
    def __init__(self, ds, balance, weights):
        super(NNBT, self).__init__(ds, balance)
        self.model = get_model()
        self.model.set_weights(weights)

    def process_bar(self):
        x_data = []
        for i in range(12):
            x_data.append(self.ds[i-11].close)
        inp = np.asanyarray(x_data)
        inp = np.expand_dims(inp, 0)

        prediction = self.model.predict(inp)[0]
        prediction = np.argmax(prediction)
        if prediction == 0:
            if not self.position:
                self.buy(self.ds[0].close, 1000.0)
        if prediction == 2:
            if self.position:
                self.sell(self.ds[0].close)


def simple_es():
    df = pd.read_csv('btc_etc.csv').rename(columns={
        'Close': 'close',
        'Date time': 'datetime',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Volume': 'volume'
    })

    @timeit
    def get_reward(weights, df):
        ds = DataSeries(df)
        bt = NNBT(ds, balance=1000.0, weights=weights)
        bt.run()
        return bt.get_profit() - 200.0

    model = get_model()

    es = EvolutionStrategy(model.get_weights(), get_reward, population_size=10, sigma=0.1, learning_rate=0.001, get_reward_func_args=[df])
    es.run(1000, print_step=1)


def integrate_stockstats():
    df = pd.read_csv('btc_etc.csv').rename(columns={
        'Close': 'close',
        'Date time': 'datetime',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Volume': 'volume'
    })
    ds = DataSeries(df)

    bt = EMABT(ds, 1000.0, 200)
    bt.run()

    print('Profit: ${:.2f}'.format(bt.get_profit()))
    # bt.plot()

if __name__ == "__main__":
    # simple_sma()
    # simple_rsi()
    # optimize_params()
    # integrate_stockstats()
    simple_es()