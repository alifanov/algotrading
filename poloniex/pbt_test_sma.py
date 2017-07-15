import pybacktest  # obviously, you should install pybacktest before importing it
import pandas as pd

ohlc = pybacktest.load_from_csv('btc_etc.csv')
ohlc.tail()

short_ma = 50
long_ma = 200

ms = ohlc.C.rolling(short_ma).mean()
ml = ohlc.C.rolling(long_ma).mean()

buy = cover = (ms > ml) & (ms.shift() < ml.shift())  # ma cross up
sell = short = (ms < ml) & (ms.shift() > ml.shift())  # ma cross down

bt = pybacktest.Backtest(locals(), 'ma_cross')

bt.summary()