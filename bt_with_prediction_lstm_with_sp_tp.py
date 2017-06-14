from datetime import datetime
import backtrader as bt
import numpy as np

from keras.models import model_from_json


class SmaCross(bt.Strategy):
    params = (
        ('pfast', 100), ('pslow', 300),
        ('stoploss', 0.01),
        ('profit_mult', 3),
        ('prdata', False),
        ('prtrade', False),
        ('prorder', False),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.order_dict = {}

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
        if order.status in [order.Margin, order.Rejected]:
            return
        elif order.status == order.Completed:
            if 'name' in order.info:
                self.broker.cancel(self.order_dict[order.ref])
                self.order = None
            else:
                if order.isbuy():
                    stop_loss = order.executed.price * (1.0 - (self.p.stoploss))
                    take_profit = order.executed.price * (1.0 + self.p.profit_mult * (self.p.stoploss))

                    sl_ord = self.sell(exectype=bt.Order.Stop,
                                       price=stop_loss)
                    sl_ord.addinfo(name="Stop")

                    tkp_ord = self.sell(exectype=bt.Order.Limit,
                                        price=take_profit)
                    tkp_ord.addinfo(name="Prof")

                    self.order_dict[sl_ord.ref] = tkp_ord
                    self.order_dict[tkp_ord.ref] = sl_ord

                elif order.issell():
                    stop_loss = order.executed.price * (1.0 + (self.p.stoploss))
                    take_profit = order.executed.price * (1.0 - 3 * (self.p.stoploss))

                    sl_ord = self.buy(exectype=bt.Order.Stop,
                                      price=stop_loss)
                    sl_ord.addinfo(name="Stop")

                    tkp_ord = self.buy(exectype=bt.Order.Limit,
                                       price=take_profit)
                    tkp_ord.addinfo(name="Prof")

                    self.order_dict[sl_ord.ref] = tkp_ord
                    self.order_dict[tkp_ord.ref] = sl_ord

                if self.p.prorder:
                    print("Open: %s %s %.2f %.2f %.2f" %
                          (order.ref,
                           self.data.num2date(order.executed.dt).date().isoformat(),
                           order.executed.price,
                           order.executed.size,
                           order.executed.comm))

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
