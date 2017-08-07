import pandas as pd
from mikasa import BT, DataSeries, RSIIndicator


class RSIBT(BT):
    def __init__(self, ds, min=20, max=80):
        super(RSIBT, self).__init__(ds)
        self.min = min
        self.max = max

    def process_bar(self):
        ds = self.ds[0]
        pre_ds = self.ds[-1]

        if not self.position:
            if ds.rsi > self.min >= pre_ds.rsi:
                self.buy(ds.close, self.balance)
        else:
            if ds.rsi < self.max <= pre_ds.rsi:
                self.sell(ds.close)


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
    for _min in range(1, 51):
        for _max in range(50, 100):
            for p in range(1, 200):
                ds = DataSeries(df,
                                index=p,
                                indicators=[
                                    RSIIndicator(p, title='rsi')
                                ])
                bt = RSIBT(ds, min=_min, max=_max)
                bt.run()
                if bt.get_profit() > best_profit:
                    best_profit = bt.get_profit()
                    best_params = (p, _min, _max)
                if p % 10 == 0:
                    print('RSI: ({}, {}, {}): Profit: ${:+.2f} | Best profit: ${:.2f} with p: {} min: {} max: {}'
                          .format(p, _min, _max,
                                  bt.get_profit(),
                                  best_profit,
                                  best_params[0],
                                  best_params[1],
                                  best_params[2]
                                  ))

# $4.68 p: 15 min: 33 max: 65