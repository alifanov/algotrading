import pandas as pd
from mikasa import BT, DataSeries, RSIIndicator


class RSIBT(BT):
    def __init__(self, ds, period=1, min=20, max=80):
        super(RSIBT, self).__init__(ds)
        self.ds.add_indicator(RSIIndicator(period, title='rsi'))
        self.min = min
        self.max = max

    def process_bar(self):
        if not self.position:
            if self.ds[0].rsi > self.min and self.ds[-1].rsi <= self.min:
                self.buy(self.ds[0].close, self.balance)
        else:
            if self.ds[0].rsi < self.max and self.ds[-1].rsi >= self.max:
                self.sell(self.ds[0].close)


if __name__ == "__main__":
    df = pd.read_csv('../datasets/btc_etc.csv').rename(columns={
        'Close': 'close',
        'Date time': 'datetime',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Volume': 'volume'
    })
    best_profit = -1000.0
    best_params = None
    ds = DataSeries(df)
    for _min in range(10, 31):
        for _max in range(70, 91):
            for p in range(1, 200):
                ds.index = p
                bt = RSIBT(ds, period=p, min=_min, max=_max)
                bt.run()
                if bt.get_profit() > best_profit:
                    best_profit = bt.get_profit()
                    best_params = (p, _min, _max)
                print('RSI: ({}, {}, {}): Profit: ${:+.2f} | Best profit: ${:.2f} with p: {} min: {} max: {}'
                      .format(p, _min, _max,
                              bt.get_profit(),
                              best_profit,
                              best_params[0],
                              best_params[1],
                              best_params[2]
                              ))
