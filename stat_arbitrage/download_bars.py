import time
import csv

from poloniex import Poloniex

PAIRS = [
    'BTC_ETH',
    'BTC_XRP',
    'BTC_XEM',
    'BTC_LTC',
    'BTC_DASH',
    'BTC_ETC',
    'BTC_STRAT',
    'BTC_XMR',
    'BTC_BTS',
    'BTC_ZEC',
    'BTC_STEEM',
]


class PoloniexApi:
    def __init__(self):
        self.api = Poloniex()

    def get_chart_data(self, pair):
        end = int(time.time())
        start = end - 60*60*24*120
        return self.api.returnChartData(pair, period=300, start=start, end=end)


if __name__ == '__main__':
    p = PoloniexApi()
    for pair in PAIRS:
        print('Download {}'.format(pair))
        data = p.get_chart_data(pair)
        with open('datasets/{}.csv'.format(pair), 'w') as f:
            writer = csv.DictWriter(f, fieldnames=['date', 'open', 'close', 'high', 'low'], extrasaction='ignore')
            writer.writeheader()
            for d in data:
                writer.writerow(d)