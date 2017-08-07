import pandas as pd
from mikasa import BT, DataSeries, SMAIndicator


class SmaCrossBT(BT):
    def process_bar(self):
        ds = self.ds[0]
        pre_ds = self.ds[-1]
        if not self.position:
            if ds.sma_slow < ds.sma_fast and pre_ds.sma_fast <= pre_ds.sma_slow:
                self.buy(ds.close, self.balance)
        else:
            if ds.sma_slow > ds.sma_fast and pre_ds.sma_fast >= pre_ds.sma_slow:
                self.sell(ds.close)


def run():
    df = pd.read_csv('../datasets/btc_etc.csv').rename(columns={
        'Close': 'close',
        'Date time': 'datetime',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Volume': 'volume'
    })
    ds = DataSeries(df, indicators=[
        SMAIndicator(1, title='sma_fast'),
        SMAIndicator(4, title='sma_slow')
    ], index=max(1, 4))
    bt = SmaCrossBT(ds)
    bt.run()


if __name__ == "__main__":
    run()
