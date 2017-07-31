import pandas as pd
from mikasa import BT, DataSeries, SMAIndicator


class SmaCrossBT(BT):
    def __init__(self, ds, period_fast=1, period_slow=2):
        super(SmaCrossBT, self).__init__(ds)
        self.ds.add_indicator(SMAIndicator(period_fast, title='sma_fast'))
        self.ds.add_indicator(SMAIndicator(period_slow, title='sma_slow'))

    def process_bar(self):
        if not self.position:
            if self.ds[0].sma_fast > self.ds[0].sma_slow and \
                            self.ds[-1].sma_fast <= self.ds[-1].sma_slow:
                self.buy(self.ds[0].close, self.balance)
        else:
            if self.ds[0].sma_fast < self.ds[0].sma_slow and \
                            self.ds[-1].sma_fast >= self.ds[-1].sma_slow:
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
    for fp in range(1, 100):
        for sp in range(1, 1000):
            ds.index = max(fp, sp)
            bt = SmaCrossBT(ds, period_fast=fp, period_slow=sp)
            bt.run()
            if bt.get_profit() > best_profit:
                best_profit = bt.get_profit()
                best_params = (fp, sp)
            print('({}, {}): Profit: ${:+.2f} | Best profit: ${:.2f} with fp: {} sp: {}'.format(fp, sp, bt.get_profit(),
                                                                                                best_profit,
                                                                                                best_params[0],
                                                                                                best_params[1]))
# fp: 4, sp: 1, profit: $7.96