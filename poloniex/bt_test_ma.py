import backtrader as bt
import backtrader.indicators as btind


class SMA_CrossOver(bt.Strategy):

    params = (('fast', 10), ('slow', 30))

    def __init__(self):

        sma_fast = btind.SMA(period=self.p.fast)
        sma_slow = btind.SMA(period=self.p.slow)

        self.buysig = btind.CrossOver(sma_fast, sma_slow)

    def next(self):
        if self.position.size:
            if self.buysig < 0:
                self.sell()

        elif self.buysig > 0:
            self.buy()


class SimpleSMAStrategy(bt.SignalStrategy):
    params = dict(
        diff=0.0000001,
        maperiod=160
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0],
            period=self.p.maperiod
        )

    def stop(self):
        self.log('(MA Period %2d, Diff: %.8f) Ending Value %.2f' %
                 (self.params.maperiod, self.p.diff, self.broker.getvalue()), doprint=True)

    def log(self, txt, dt=None, doprint=False):
        if doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def next(self):
        if not self.position:
            if self.dataclose[0] > self.sma[0] + self.p.diff:
                self.buy()

        else:
            if self.dataclose[0] < self.sma[0] - self.p.diff:
                self.sell()

cerebro = bt.Cerebro()

# strats = cerebro.optstrategy(
#         SimpleSMAStrategy,
#         maperiod=range(1, 200),
#         diff=[0.1, 0.01, 0.001, 0.0001, 0.00001, 0.0000001, 0.00000001]
# )

data = bt.feeds.GenericCSVData(
    dataname='btc_etc.csv',
    separator=',',
    dtformat=('%Y-%m-%d %H:%M:%S'),
    datetime=2,
    open=5,
    high=3,
    low=4,
    close=1,
    volume=7,
    openinterest=-1
)

cerebro.broker.setcash(1000.0)
cerebro.adddata(data)
cerebro.addsizer(bt.sizers.FixedSize, stake=100)

cerebro.addstrategy(SimpleSMAStrategy)

# Print out the starting conditions
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Run over everything
cerebro.run()

# Print out the final result
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

cerebro.plot()
