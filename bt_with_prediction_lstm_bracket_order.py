from datetime import datetime, timedelta
import backtrader as bt
import numpy as np

from keras.models import model_from_json


class SmaCross(bt.Strategy):
    params = dict(
        limit=0.005,
        limdays=2,
        limdays2=1000,
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.orefs = list()

        # self.signal_add(bt.SIGNAL_LONG, bt.ind.CrossOver(sma1, sma2))
        # load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("model.h5")
        print("Loaded model from disk")

        # evaluate loaded model on test data
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.getstatusname() not in ['Accepted', 'Submitted']:
            print('{}: Order ref: {} / Type: {} / Status: {} / Price: {:.4f}'.format(
                self.data.datetime.date(0),
                order.ref, 'Buy' * order.isbuy() or 'Sell',
                order.getstatusname(),
                order.price
            ))

        if not order.alive() and order.ref in self.orefs:
            self.orefs.remove(order.ref)

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        predicted_close = self.model.predict(np.array([[[self.dataclose[0]]]]))
        predicted_close = predicted_close[0][0]
        prev_predicted_close = self.model.predict(np.array([[[self.dataclose[-1]]]]))
        prev_predicted_close = prev_predicted_close[0][0]

        if self.orefs:
            return  # pending orders do nothing

        if not self.position:
            if predicted_close > prev_predicted_close:
                close = self.data.close[0]
                p1 = close * (1.0 - self.p.limit)
                p2 = p1 - 0.02 * close
                p3 = p1 + 0.06 * close
                print('p1: {:.4f}, p2: {:.4f}, p3: {:.4f}'.format(
                    p1,
                    p2,
                    p3
                ))

                valid1 = timedelta(self.p.limdays)
                valid2 = valid3 = timedelta(self.p.limdays2)

                os = self.buy_bracket(
                    price=p1, valid=valid1,
                    stopprice=p2, stopargs=dict(valid=valid2),
                    limitprice=p3, limitargs=dict(valid=valid3), )

                self.orefs = [o.ref for o in os]

            if predicted_close < prev_predicted_close:
                close = self.data.close[0]
                p1 = close * (1.0 + self.p.limit)
                p2 = p1 + 0.02 * close
                p3 = p1 - 0.06 * close
                print('p1: {:.4f}, p2: {:.4f}, p3: {:.4f}'.format(
                    p1,
                    p2,
                    p3
                ))

                valid1 = timedelta(self.p.limdays)
                valid2 = valid3 = timedelta(self.p.limdays2)

                os = self.sell_bracket(
                    price=p1, valid=valid1,
                    stopprice=p2, stopargs=dict(valid=valid2),
                    limitprice=p3, limitargs=dict(valid=valid3), )

                self.orefs = [o.ref for o in os]

        # if not self.position:
        #     if predicted_close > prev_predicted_close:
        #         self.order = self.buy()
        # else:
        #     if predicted_close < prev_predicted_close:
        #         self.order = self.sell()

cerebro = bt.Cerebro()

data = bt.feeds.GenericCSVData(
    dataname='eur_usd_1d.csv',
    separator=',',
    dtformat=('%Y%m%d'),
    tmformat=('%H%M%S'),
    datetime=0,
    time=-1,
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
cerebro.broker.setcash(1000.0)

cerebro.addsizer(bt.sizers.FixedSize, stake=100)

cerebro.addstrategy(SmaCross)
# Print out the starting conditions
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Run over everything
cerebro.run()

# Print out the final result
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

# cerebro.plot()
