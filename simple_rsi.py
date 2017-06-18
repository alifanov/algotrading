from datetime import datetime, timedelta
import backtrader as bt


class RSIStrategy(bt.SignalStrategy):
    params = dict(
        diff=0.01,
      limit=0.005,
      limdays=10,
      limdays2=1000,
        period=14,
        rsi_top=70,
        rsi_bottom=30
    )

    def __init__(self):
        self.order = None
        self.dataclose = self.datas[0].close
        self.rsi = bt.indicators.RelativeStrengthIndex(
            period=self.p.period
        )

    def log(self, txt, dt=None, doprint=False):
        '''Logging function fot this strategy'''
        if doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def stop(self):
        cash = self.broker.getvalue()
        if cash > 10000:
            self.log('(RSI period: %d [%d, %d]) Ending Value %.2f' %
                     (self.p.period, self.p.rsi_bottom, self.p.rsi_top, self.broker.getvalue()), doprint=True)

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.rsi[0] > self.p.rsi_bottom and self.rsi[-1] <= self.p.rsi_bottom:
                self.order = self.buy()
        else:
            if self.rsi[0] < self.p.rsi_top and self.rsi[-1] >= self.p.rsi_top:
                self.order = self.sell()

cerebro = bt.Cerebro()

strats = cerebro.optstrategy(
        RSIStrategy,
    rsi_top=range(60, 100),
    rsi_bottom=range(1, 41),
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
