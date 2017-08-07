import pandas as pd
from mikasa import BT, DataSeries, MomentumIndicator


class MomentumBT(BT):
    def process_bar(self):
        ds = self.ds[0]
        pre_ds = self.ds[-1]
        pre_pre_ds = self.ds[-2]

        if not self.position:
            if pre_ds.momentum < ds.momentum <= pre_pre_ds.momentum:
                self.buy(ds.close, self.balance)
        else:
            if pre_ds.momentum > ds.momentum >= pre_pre_ds.momentum:
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
    for p in range(1, 200):
        ds = DataSeries(df,
                        index=max(p, 2),
                        indicators=[
                            MomentumIndicator(p)
                        ])
        bt = MomentumBT(ds)
        bt.run()
        if bt.get_profit() > best_profit:
            best_profit = bt.get_profit()
            best_params = (p, )
        if p % 10 == 0:
            print('Momentum: ({}, ): Profit: ${:+.2f} | Best profit: ${:.2f} with p: {}'
                  .format(p,
                          bt.get_profit(),
                          best_profit,
                          best_params[0],
                          ))
