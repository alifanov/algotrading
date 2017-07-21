import numpy as np
import pandas as pd

from mikasa import BT, DataSeries
from lstm_train import get_loaded_model
from sklearn.externals import joblib


class NNBT(BT):
    def __init__(self, ds, balance, model, look_back=1, scaler=None):
        super(NNBT, self).__init__(ds, balance)
        self.model = model
        self.look_back = look_back
        self.scaler = scaler

    def get_prediction(self):
        x_data = []
        for i in range(self.look_back):
            x_data.append(self.ds[i - self.look_back].close)
        inp = np.asanyarray(x_data)
        inp = inp.reshape(-1, 1)
        inp = np.expand_dims(inp, 0)

        prediction = self.model.predict(inp)[0]
        prediction = self.scaler.inverse_transform(prediction)[0]
        return prediction

    def process_bar(self):
        prediction = self.get_prediction()

        if prediction > self.ds[0].close:
            if not self.position:
                self.buy(self.ds[0].close, 1000.0)
        if prediction < self.ds[0].close:
            if self.position:
                self.sell(self.ds[0].close)


def backtesting_with_lstm():
    model = get_loaded_model()
    df = pd.read_csv('btc_etc.csv').rename(columns={
        'Close': 'close',
        'Date time': 'datetime',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Volume': 'volume'
    })
    ds = DataSeries(df)

    scaler = joblib.load(open('scaler.sav', 'rb'))
    look_back = 1

    bt = NNBT(ds, 1000.0, model, look_back, scaler)
    bt.run()

    print('Profit: ${:.2f}'.format(bt.get_profit()))


backtesting_with_lstm()
