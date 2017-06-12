from datetime import datetime
import backtrader as bt
import numpy as np

from keras.models import model_from_json


class SmaCross(bt.Strategy):
    params = (('pfast', 100), ('pslow', 300),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None

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
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enougth cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, Price: %.3f, Cost: %.3f, Comm %.3f, Size: %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm,
                          order.executed.size))
            elif order.issell():
                self.log('SELL EXECUTED, Price: %.3f, Cost: %.3f, Comm %.3f, Size: %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm,
                          order.executed.size))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        # print('Date: ', self.datas[0].datetime.datetime(0))
        predicted_close = self.model.predict(np.array([[[self.dataclose[0]]]]))
        # predicted_close = scaler.inverse_transform(predicted_close)[0][0]
        predicted_close = predicted_close[0][0]
        prev_predicted_close = self.model.predict(np.array([[[self.dataclose[-1]]]]))
        prev_predicted_close = prev_predicted_close[0][0]
        # print('Current close price: ', self.dataclose[0])
        # print('Predicted next close price: ', predicted_close)

        if self.order:
            return

        if not self.position:
            if predicted_close > prev_predicted_close:
                self.order = self.buy()
        else:
            if predicted_close < prev_predicted_close:
                self.order = self.sell()

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

cerebro.addsizer(bt.sizers.FixedSize, stake=10)

cerebro.addstrategy(SmaCross)
# Print out the starting conditions
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Run over everything
cerebro.run()

# Print out the final result
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

# cerebro.plot()
