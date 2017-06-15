from datetime import datetime, timedelta
import backtrader as bt


class SimpleSMAStrategy(bt.SignalStrategy):
    params = dict(
        diff=0.01,
      limit=0.005,
      limdays=10,
      limdays2=1000,
      maperiod=30)

    def __init__(self):
        self.orefs = list()
        self.dataclose = self.datas[0].close
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0],
            period=self.params.maperiod
        )

    def notify_order(self, order):
        # if order.getstatusname() not in ['Accepted', 'Submitted']:
            # print('{}: Order ref: {} / Type: {} / Status: {} / Price: {:.4f}'.format(
            #     self.data.datetime.date(0),
            #     order.ref, 'Buy' * order.isbuy() or 'Sell',
            #     order.getstatusname(),
            #     order.price
            # ))

        if not order.alive() and order.ref in self.orefs:
            self.orefs.remove(order.ref)

    def log(self, txt, dt=None, doprint=False):
        ''' Logging function fot this strategy'''
        if doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def stop(self):
        self.log('(MA Period %2d) Ending Value %.2f' %
                 (self.params.maperiod, self.broker.getvalue()), doprint=True)

    def next(self):
        if self.orefs:
            return  # pending orders do nothing

        if not self.position:
            if self.dataclose[0] > self.sma[0]:
                close = self.data.close[0]
                p1 = close * (1.0 - self.p.limit)
                p2 = p1 - self.p.diff * close
                p3 = p1 + self.p.diff * close
                # print('start price: {:.4f}, stop loss: {:.4f}, take profit: {:.4f}'.format(
                #     p1,
                #     p2,
                #     p3
                # ))

                valid1 = timedelta(self.p.limdays)
                valid2 = valid3 = timedelta(self.p.limdays2)

                os = self.buy_bracket(
                    price=p1, valid=valid1,
                    stopprice=p2, stopargs=dict(valid=valid2),
                    limitprice=p3, limitargs=dict(valid=valid3), )

                self.orefs = [o.ref for o in os]
        else:
            if self.dataclose[0] < self.sma[0]:
                close = self.data.close[0]
                p1 = close * (1.0 + self.p.limit)
                p2 = p1 + self.p.diff * close
                p3 = p1 - self.p.diff * close
                # print('start price: {:.4f}, stop loss: {:.4f}, take profit: {:.4f}'.format(
                #     p1,
                #     p2,
                #     p3
                # ))

                valid1 = timedelta(self.p.limdays)
                valid2 = valid3 = timedelta(self.p.limdays2)

                os = self.sell_bracket(
                    price=p1, valid=valid1,
                    stopprice=p2, stopargs=dict(valid=valid2),
                    limitprice=p3, limitargs=dict(valid=valid3), )

                self.orefs = [o.ref for o in os]

cerebro = bt.Cerebro()

strats = cerebro.optstrategy(
        SimpleSMAStrategy,
        maperiod=range(2, 100))

data = bt.feeds.GenericCSVData(
    dataname='eur_usd_15m.csv',
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
cerebro.addsizer(bt.sizers.FixedSize, stake=10)

# cerebro.addstrategy(SimpleSMAStrategy)
# Print out the starting conditions
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Run over everything
cerebro.run()

# Print out the final result
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

# cerebro.run()
# cerebro.plot()
