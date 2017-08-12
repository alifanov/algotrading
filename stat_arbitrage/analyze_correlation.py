import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


from download_bars import PAIRS

dfs = []

for pair in PAIRS:
    df = pd.read_csv(open('datasets/{}.csv'.format(pair)), index_col='date').rename(columns={'close': pair})
    df.drop('open', 1, inplace=True)
    df.drop('high', 1, inplace=True)
    df.drop('low', 1, inplace=True)
    df = (df-df.mean())/df.std()
    dfs.append(df)

result = pd.concat(dfs, axis=1)
corr = result.corr()

print('Top correlations: ')
corr = corr.abs().unstack()
print(corr[corr != 1.0].sort_values().tail(10))

# result[['BTC_XMR', 'BTC_XEM']].plot()

# diff
# diff = result['BTC_XMR'] - result['BTC_XEM']
# diff.plot()

# sns.heatmap(corr, xticklabels=result.columns, yticklabels=result.columns, annot=True, cmap='RdYlGn_r')
# plt.show()

# BTC_XRP, BTC_ETC
# BTC_LTC, BTC_ETC