from mikasa import *


class LastCloseBT(BT):
    def process_bar(self):
        if not self.position:
            current_diff = self.ds[0].close - self.ds[-1].close
            prev_diff = self.ds[-1].close - self.ds[-2].close
            prev_prev_diff = self.ds[-2].close - self.ds[-3].close
            if prev_prev_diff < 0 and prev_diff > 0 and current_diff > 0 and self.ds[0].volume > 5.0:
                self.buy(self.ds[0].close, 1000.0)
        else:
            if self.ds[0].close - self.ds[-1].close < 0:
                self.sell(self.ds[0].close)


def validate_close():
    df = pd.read_csv('btc_etc.csv').rename(columns={
        'Close': 'close',
        'Date time': 'datetime',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Volume': 'volume'
    })
    ds = DataSeries(df)

    bt = LastCloseBT(ds, 1000.0)
    bt.run()

    print('Profit: ${:.2f}'.format(bt.get_profit()))


if __name__ == "__main__":
    validate_close()
