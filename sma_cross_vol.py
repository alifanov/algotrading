from datetime import datetime, timedelta
import backtrader as bt


class SMACrossVolumeStrategy(bt.SignalStrategy):
    params = dict(
        diff=0.01,
      limit=0.005,
      limdays=10,
      limdays2=1000,
      maperiod_small=30,
      maperiod_big=30,
    )

    def __init__(self):
        self.order = None
        self.dataclose = self.datas[0].close
        self.datavol = self.datas[0].volume
        self.sma_small = bt.indicators.SimpleMovingAverage(
            self.datas[0],
            period=self.params.maperiod_small
        )
        self.sma_big = bt.indicators.SimpleMovingAverage(
            self.datas[0],
            period=self.params.maperiod_big
        )

    def log(self, txt, dt=None, doprint=False):
        '''Logging function fot this strategy'''
        if doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def stop(self):
        self.log('(MA Period Small: %2d | MA Period Big: %2d) Ending Value %.2f' %
                 (self.p.maperiod_small, self.p.maperiod_big, self.broker.getvalue()), doprint=True)

    def next(self):
        if self.order:
            return
        if not self.position:
            if self.sma_small[0] > self.sma_big[0] and self.sma_small[-1] < self.sma_big[-1] and self.datavol[0] > 2000000:
                self.order = self.buy()
        else:
            if self.sma_small[0] < self.sma_big[0] and self.sma_small[-1] > self.sma_big[-1] and self.datavol[0] > 2000000:
                self.order = self.sell()

cerebro = bt.Cerebro()

strats = cerebro.optstrategy(
    SMACrossVolumeStrategy,
        maperiod_small=range(2, 10),
        maperiod_big=range(10, 20),
)

data = bt.feeds.GenericCSVData(
    dataname='eur_usd_1d.csv',
    separator=',',
    dtformat=('%Y%m%d'),
    tmformat=('%H%M00'),
    datetime=0,
    time=1,
    open=2,
    high=3,
    low=4,
    close=5,
    volume=6,
    openinterest=-1
)

# data = bt.feeds.YahooFinanceData(dataname='YHOO', fromdate=datetime(2011, 1, 1),
#                                  todate=datetime(2012, 12, 31))
cerebro.adddata(data)
cerebro.addsizer(bt.sizers.FixedSize, stake=50)

# cerebro.addstrategy(SimpleSMAStrategy)
# Print out the starting conditions
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Run over everything
cerebro.run()

# Print out the final result
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

# cerebro.run()
# cerebro.plot()
