from datetime import datetime, timedelta
import time
import requests
import pandas as pd

import matplotlib.pyplot as plt


class Poloniex():
    PUBLIC_URL = 'https://poloniex.com/public'

    def returnChartData(self, currencyPair, period, start, end):
        r = requests.get(self.PUBLIC_URL, {
            'command': 'returnChartData',
            'currencyPair': currencyPair,
            'period': period,
            'start': start,
            'end': end
        })
        return r.json()

polo = Poloniex()

start_time = time.mktime((datetime.now() - timedelta(days=7)).timetuple())
end_time = time.mktime((datetime.now()).timetuple())

charts = polo.returnChartData('BTC_ETC', period=300, start=start_time, end=end_time)
df = pd.DataFrame(charts)
close_df = df[['date', 'close']]
close_df.set_index(['date'], inplace=True)
close_df['emwa'] = pd.ewma(close_df, span=10, min_periods=10)
close_df.plot()
plt.show()