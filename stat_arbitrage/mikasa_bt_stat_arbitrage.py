import pandas as pd
from mikasa import BT, DataSeries, SMAIndicator

import matplotlib.pyplot as plt
import seaborn as sns


class StatArbitrageBT(BT):
    def __init__(self, datas, balance, limit):
        self.limit = limit
        super(StatArbitrageBT, self).__init__(datas, balance)

    def process_bar(self):
        d0 = self.datas[0]
        d1 = self.datas[1]
        d2 = self.datas[2]
        if not self.position:
            if d2[0].close > self.limit:
                self.buy(d1[0].close, self.balance)
        else:
            if d2[0].close < 0.0001:
                self.sell(d1[0].close)

df0 = pd.read_csv('datasets/BTC_ETH.csv').rename(columns={'date': 'datetime'})
# df0_scaled = (df0 - df0.mean()) / df0.std()

df1 = pd.read_csv('datasets/BTC_ZEC.csv').rename(columns={'date': 'datetime'})
# df1_scaled = (df1 - df1.mean()) / df1.std()

df2 = df0 - df1
# df2 = (df2 - df2.mean()) / df2.std()

df2['close'].plot()
plt.axhline(df2.mean().close)
plt.axhline(df2.mean().close + df2.std().close)
plt.axhline(df2.mean().close - df2.std().close)


# ds0 = DataSeries(df0)
# ds1 = DataSeries(df1)
# ds2 = DataSeries(df2)

# profit = []

# for limit in range(0, 1000):
#     ds0.index = 0
#     ds1.index = 0
#     ds2.index = 0
#
#     bt = StatArbitrageBT([ds0, ds1, ds2], balance=1000.0, limit=limit/1000.0)
#     bt.run()
#
#     profit.append(bt.get_profit())
#
#     print('Limit: {:.3f}'.format(limit/1000.0))
#     print('Profit: ${:.2f}'.format(bt.get_profit()))
#     print('ROI: ${:.2%}'.format(bt.get_roi()))
#     print()
#
# profit = pd.Series(profit)
# profit.plot()
plt.show()