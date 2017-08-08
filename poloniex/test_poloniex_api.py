import hashlib
import hmac
import config
import time
import requests

from poloniex import Poloniex
from urllib.parse import urlencode

MIN_SPREAD = 1e-4
TRADE_VOLUME = 1e-4 + 1e-5

polo = Poloniex()


class YobitAPI:
    TRADE_API = 'https://yobit.net/tapi'

    def __init__(self, api_key, secret):
        self.api_key = api_key
        self.secret = secret

    def api_call(self, method, params=None):
        if params is None:
            params = {}
        params['method'] = method
        params['nonce'] = str(int(time.time()))
        post_data = urlencode(params).encode()
        signature = hmac.new(
            self.secret.encode(),
            post_data,
            hashlib.sha512).hexdigest()
        headers = {
            'Sign': signature,
            'Key': self.api_key,
            'User-Agent': "Mozilla/5.0"
        }
        response = requests.post(YobitAPI.TRADE_API, data=params, headers=headers)
        response.raise_for_status()
        return response.json()

    def get_active_orders(self, pair):
        return self.api_call('ActiveOrders', {'pair': pair})

    def get_info(self):
        return self.api_call('getInfo')

    def trade(self, pair, type, price, volume):
        if type not in ['buy', 'sell']:
            raise Exception('type should be in ["buy", "sell"]')
        return self.api_call('Trade', {
            'pair': pair,
            'type': type,
            'rate': price,
            'amount': volume
        })

    def withdraw(self, coin, volume, address):
        return self.api_call('WithdrawCoinsToAddress', {
            'coinName': coin,
            'amount': volume,
            'address': address
        })


def get_min_order(orders):
    result_vol = 0
    result_price = 1e6
    for o in orders:
        o = list(map(float, o))
        if o[0] < result_price:
            result_price = o[0]
            result_vol = o[1]
    return result_price, result_vol


def get_max_order(orders):
    result_vol = 0
    result_price = 0
    for o in orders:
        o = list(map(float, o))
        if o[0] > result_price:
            result_price = o[0]
            result_vol = o[1]
    return result_price, result_vol


def get_ticker(pair):
    return polo.returnTicker()[pair]


def get_orders(pair):
    return polo.returnOrderBook()[pair]


def arbitrage(pair):
    orders = get_orders(pair)
    highest_bid, bid_vol = get_max_order(orders['bids'])
    sell_price = float(highest_bid)

    _pair = map(str.lower, pair.split('_')[::-1])
    _pair = list(_pair)
    _target_coin, _base_coin = _pair
    _pair = '_'.join(_pair)
    r = requests.get('https://yobit.net/api/3/depth/{}'.format(_pair))
    r.raise_for_status()
    data = r.json()

    orders = data[_pair]
    lowest_ask, ask_vol = get_min_order(orders['asks'])
    buy_price = float(lowest_ask)

    spread = sell_price - buy_price
    print(spread)
    if spread > MIN_SPREAD:
        print('Spread:\t{:.6f}\tVolume:\t{:.6f}'.format(spread, min(ask_vol, bid_vol)))
        yoapi = YobitAPI(config.API_KEY, config.API_SECRET)
        print(yoapi.trade(_pair, 'buy', buy_price, TRADE_VOLUME))
        print('Buy ({}): {} for {}'.format(_pair, TRADE_VOLUME, buy_price))
        # yoapi.withdraw(_target_coin, TRADE_VOLUME, config.TARGET_LTC_ADDRESS)
        # print('Transfer {} {} to {}'.format(TRADE_VOLUME, _target_coin, config.TARGET_ETH_ADDRESS))

        # polo.sell(pair, sell_price, TRADE_VOLUME)
        # print('Sell ({}): {} for {}'.format(pair, TRADE_VOLUME, sell_price))
        # balance = polo.returnBalances()[_base_coin]
        # polo.withdraw(_base_coin, balance, config.TARGET_BTC_ADDRESS)
        # print('Transfer {} {} to {}'.format(balance, _base_coin, config.TARGET_BTC_ADDRESS))


if __name__ == "__main__":
    arbitrage('BTC_ETH')
    # while True:
    #     for pair in ['BTC_ETH']:
    #         orders = get_orders(pair)
    #         highest_bid, bid_vol = get_max_order(orders['bids'])
    #         sell_price = float(highest_bid)
    #
    #         _pair = map(str.lower, pair.split('_')[::-1])
    #         _pair = '_'.join(_pair)
    #         r = requests.get('https://yobit.net/api/3/depth/{}'.format(_pair))
    #         r.raise_for_status()
    #         data = r.json()
    #
    #         orders = data[_pair]
    #         lowest_ask, ask_vol = get_min_order(orders['asks'])
    #         buy_price = float(lowest_ask)
    #
    #         print('Spread:\t{:.6f}\tVolume:\t{:.6f}'.format(sell_price - buy_price, min(ask_vol, bid_vol)))
    #     time.sleep(3.0)
