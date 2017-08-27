import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod

from event import SignalEvent
from utils import rolling_beta


class Strategy(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def calculate_signals(self, queue):
        raise NotImplementedError("Should implement calculate_signals()")


class BuyAndHoldStrategy(Strategy):
    def __init__(self, bars, queue):
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.queue = queue

        # Once buy & hold signal is given, these are set to True
        self.bought = self._calculate_initial_bought()

    def _calculate_initial_bought(self):
        bought = {}
        for s in self.symbol_list:
            bought[s] = False
        return bought

    def calculate_signals(self, event):
        if event.type == 'MARKET':
            for s in self.symbol_list:
                bars = self.bars.get_latest_bars(s, N=1)
                if bars is not None and bars != []:
                    if not self.bought[s]:
                        # (Symbol, Datetime, Type = LONG, SHORT or EXIT)
                        signal = SignalEvent(bars[0][0], bars[0][1], 'LONG', 1.0)
                        self.queue.put(signal)
                        self.bought[s] = True


class StatArbitrageStrategy(Strategy):
    LOOK_BACK = 20
    Z_ENTRY_THRESHOLD = 2.0
    Z_EXIT_THRESHOLD = 1.0

    def __init__(self, bars, queue):
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.queue = queue

    def calculate_signals(self, event):
        if event.type == 'MARKET':
            s0 = 'BTC_ETC'
            s1 = 'BTC_LTC'
            if len(self.bars.latest_symbol_data[s0]) >= self.LOOK_BACK and len(
                    self.bars.latest_symbol_data[s1]) > self.LOOK_BACK:

                X = self.bars.get_latest_bars(s0, self.LOOK_BACK)
                Y = self.bars.get_latest_bars(s1, self.LOOK_BACK)

                X = pd.DataFrame(X)
                X.set_index('datetime', inplace=True)
                Y = pd.DataFrame(Y)
                Y.set_index('datetime', inplace=True)

                print(X)

                ols = rolling_beta(X['close'], Y['close'], X.index, window=self.LOOK_BACK)
                pairs = pd.DataFrame(X['close'], Y['close'], columns=['{}_close'.format(s0), '{}_close'.format(s1)])

                pairs['hedge_ratio'] = ols['beta']
                pairs['hedge_ratio'] = [v for v in pairs['hedge_ratio'].values]
                pairs['spread'] = pairs['{}_close'.format(s0)] - pairs['hedge_ratio'] * pairs['{}_close'.format(s1)]
                pairs['zscore'] = (pairs['spread'] - np.mean(pairs['spread'])) / np.std(pairs['spread'])

                if pairs['zscore', -1] <= -self.Z_ENTRY_THRESHOLD:
                    signal = SignalEvent(s0, X['close', -1], 'LONG', 1.0)
                    self.queue.put(signal)
                    signal = SignalEvent(s1, Y['close', -1], 'SHORT', 1.0)
                    self.queue.put(signal)
                elif pairs['zscore', -1] >= self.Z_ENTRY_THRESHOLD:
                    signal = SignalEvent(s0, X['close', -1], 'SHORT', 1.0)
                    self.queue.put(signal)
                    signal = SignalEvent(s1, Y['close', -1], 'LONG', 1.0)
                    self.queue.put(signal)
                elif pairs['zscore', -1] <= self.Z_EXIT_THRESHOLD:
                    signal = SignalEvent(s0, X['close', -1], 'EXIT', 1.0)
                    self.queue.put(signal)
                    signal = SignalEvent(s1, Y['close', -1], 'EXIT', 1.0)
                    self.queue.put(signal)
