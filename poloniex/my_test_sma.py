import pandas as pd
import matplotlib.pyplot as plt


class SMAIndicator:
    title = 'sma'

    def __init__(self, period, title='sma'):
        self.period = period
        self.title = title

    def get_data(self, ds):
        return ds.rolling(center=False, window=self.period).mean()


class MomentumIndicator:
    def __init__(self, period=14, title='momentum'):
        self.period = period
        self.title = title

    def get_data(self, ds):
        if len(ds) > self.period - 1:
            return ds[-1] * 100 / ds[-self.period]
        return None


class DataSeries:
    def __init__(self, data, indicators=[]):
        self.data = data
        self.index = 0
        self.indicators = indicators
        for indicator in self.indicators:
            self.data[indicator.title] = indicator.get_data(self.data['close'])

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
    def __init__(self, data_path, balance=1000.0):
        self.ds = DataSeries(pd.read_csv(data_path).rename(columns={
            'Close': 'close',
            'Date time': 'datetime',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Volume': 'volume'
        }), indicators=[SMAIndicator(period=21)])
        self.balance = balance
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
        # print('SELL: price {} with profit {}'.format(price, self.position.get_profit()))
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

    def run(self):
        for _ in self.ds:
            self.process_bar()

    def plot(self):
        data = self.ds.data
        data = data.set_index('datetime')
        headers = ['close']
        for ind in self.ds.indicators:
            headers.append(ind.title)
        data = data[headers]
        data.plot()

        trade_open_datetimes = [trade.open_datetime for trade in self.trades]
        trade_open_prices = [trade.open_price for trade in self.trades]
        plt.plot(trade_open_datetimes, trade_open_prices, 'g^')

        trade_close_datetimes = [trade.close_datetime for trade in self.trades]
        trade_close_prices = [trade.close_price for trade in self.trades]
        plt.plot(trade_close_datetimes, trade_close_prices, 'rv')

        plt.show()

if __name__ == "__main__":
    bt = BT('btc_etc.csv')
    start_balance = bt.balance
    bt.run()
    end_balance = bt.balance
    profit = end_balance - start_balance
    roi = 1.0*profit/start_balance

    print('Start balance:\t${:.2f}'.format(start_balance))
    print('End balance:\t${:.2f}'.format(end_balance))
    print('Profit:\t\t${:.2f} ({:+.2%})'.format(profit, roi))

    # bt.plot()
