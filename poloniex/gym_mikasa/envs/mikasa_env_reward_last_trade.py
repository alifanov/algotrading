import pandas as pd
import gym
from gym import error, spaces, utils

from mikasa import *

ACTION_LOOKUP = {
    0: 'nop',
    1: 'buy',
    2: 'sell'
}


class MikasaRewardLastTradeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1000.0, shape=(6, ))

    def _get_reward(self):
        return self.bt.get_profit()

    def _step(self, action):
        reward = self._take_action(action)
        ob = self._get_observation()
        episode_over = self.ds.is_end()
        return ob, reward, episode_over, {}

    def _take_action(self, action):
        reward = 0.0
        if ACTION_LOOKUP[action] == 'buy' and not self.bt.position:
            self.bt.buy(self.ds[0].close, self.balance)
        if ACTION_LOOKUP[action] == 'sell' and self.bt.position:
            self.bt.sell(self.ds[0].close)
            last_trade = self.bt.trades[-1]
            reward = last_trade.get_profit()
        if not self.bt.ds.is_end():
            self.bt.go()
        return reward

    def _reset(self):
        df = pd.read_csv('btc_etc_first100.csv').rename(columns={
            'Close': 'close',
            'Date time': 'datetime',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Volume': 'volume'
        })
        self.balance = 1000.0
        self.ds = DataSeries(df)
        self.bt = BT(self.ds, self.balance)
        return self._get_observation()

    def _get_observation(self):
        d = self.ds[0]
        return (d.open, d.high, d.low, d.close, d.volume, 1 if self.bt.position is None else 0)

    def _render(self, mode='human', close=False):
        pass
