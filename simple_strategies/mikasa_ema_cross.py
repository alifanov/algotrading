import pandas as pd
from mikasa import BT, DataSeries, BaseIndicator


class EMAIndicator(BaseIndicator):
    def __init__(self, period=14, title='ema'):
        super(EMAIndicator, self).__init__(title)
        self.period = period

    def get_data(self, df, field_name):
        return df[field_name].ewm(span=self.period, ignore_na=False, adjust=True, min_periods=self.period).mean()


class EmaCrossBT(BT):
    def __init__(self, ds, period_fast=1, period_slow=2):
        super(EmaCrossBT, self).__init__(ds)
        self.ds.add_indicator(EMAIndicator(period_fast, title='ema_fast'))
        self.ds.add_indicator(EMAIndicator(period_slow, title='ema_slow'))

    def process_bar(self):
        if not self.position:
            if self.ds[0].ema_fast > self.ds[0].ema_slow and \
                            self.ds[-1].ema_fast <= self.ds[-1].ema_slow:
                self.buy(self.ds[0].close, self.balance)
        else:
            if self.ds[0].ema_fast < self.ds[0].ema_slow and \
                            self.ds[-1].ema_fast >= self.ds[-1].ema_slow:
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
            bt = EmaCrossBT(ds, period_fast=fp, period_slow=sp)
            bt.run()
            if bt.get_profit() > best_profit:
                best_profit = bt.get_profit()
                best_params = (fp, sp)
            print('EMA Cross: ({}, {}): Profit: ${:+.2f} | Best profit: ${:.2f} with fp: {} sp: {}'.format(fp, sp,
                                                                                                           bt.get_profit(),
                                                                                                           best_profit,
                                                                                                           best_params[
                                                                                                               0],
                                                                                                           best_params[
                                                                                                               1]))
