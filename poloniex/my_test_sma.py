import pandas as pd
import matplotlib.pyplot as plt
from stockstats import StockDataFrame


class SMAIndicator:
    title = 'sma'

    def __init__(self, period, title='sma'):
        self.period = period
        self.title = title

    def get_data(self, df, field_name):
        ds = df[field_name]
        return ds.rolling(center=False, window=self.period).mean()


class MomentumIndicator:
    def __init__(self, period=14, title='momentum'):
        self.period = period
        self.title = title

    def get_data(self, df, field_name):
        ds = df[field_name]
        if len(ds) > self.period - 1:
            return ds[-1] * 100 / ds[-self.period]
        return None


class RSIIndicator:
    def __init__(self, period=14, title='rsi'):
        self.period = period
        self.title = title

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


class EMAIndicator:
    def __init__(self, period, title='ema'):
        self.period = period
        self.title = title

    def get_data(self, df, field_name):
        return df['{}_{}_ema'.format(field_name, self.period)]


class DataSeries:
    def __init__(self, data):
        self.data = StockDataFrame.retype(data.copy())
        self.index = 0
        self.indicators = []

    def add_indicator(self, indicator):
        self.data[indicator.title] = indicator.get_data(self.data, 'close')
        self.indicators.append(indicator)

    def __getitem__(self, index):
        return self.data.iloc[index + self.index]

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        value = self.data.iloc[self.index]
        self.index += 1
        if self.index >= self.data.shape[0]:
            raise StopIteration
        return value


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
        # print('Price: {} SMA: {}'.format(self.ds[0].close, self.ds[0].sma))
        if not self.position:
            current_diff = self.ds[0].close - self.ds[-1].close
            prev_diff = self.ds[-1].close - self.ds[-2].close
            prev_prev_diff = self.ds[-2].close - self.ds[-3].close
            if prev_prev_diff < 0 and prev_diff > 0 and current_diff > 0 and self.ds[0].volume > 5.0:
                # print('BUY: price {} with {} shares on volume {}'.format(self.ds[0].close, 1000.0, self.ds[0].volume))
                self.buy(self.ds[0].close, 1000.0)
        else:
            if self.ds[0].close - self.ds[-1].close < 0:
                self.sell(self.ds[0].close)

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
        for ind in self.ds.indicators:
            headers.append(ind.title)
        data = data[headers]
        data.plot()
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
    bt.plot()

if __name__ == "__main__":
    # simple_sma()
    # optimize_params()
    integrate_stockstats()