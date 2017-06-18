from datetime import datetime, timedelta
import backtrader as bt


class BBStrategy(bt.SignalStrategy):
    params = dict(
        diff=0.01,
      limit=0.005,
      limdays=10,
      limdays2=1000,
        period=14,
        devfactor=2,
    )

    def __init__(self):
        self.order = None
        self.dataclose = self.datas[0].close
        self.bb = bt.indicators.BollingerBands()

    def log(self, txt, dt=None, doprint=False):
        '''Logging function fot this strategy'''
        if doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def stop(self):
        cash = self.broker.getvalue()
        self.log('(BB period: %d | BB devfactor %d) Ending Value %.2f' %
                     (self.p.period, self.p.devfactor, self.broker.getvalue()), doprint=True)

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.dataclose[0] > self.bb.lines.bot[0] and self.dataclose[-1] <= self.bb.lines.bot[-1]:
                self.order = self.buy()
        else:
            if self.dataclose[0] < self.bb.lines.top[0] and self.dataclose[-1] >= self.bb.lines.top[-1]:
                self.order = self.sell()

cerebro = bt.Cerebro()

strats = cerebro.optstrategy(
        BBStrategy,
        devfactor=range(2, 10),
        period=range(2, 100),
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
