import pandas as pd
import gym
from gym import error, spaces, utils

from mikasa import *

ACTION_LOOKUP = {
    0: 'nop',
    1: 'buy',
    2: 'sell'
}


class MikasaLast4Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1.0, shape=(17, ))

    def _get_reward(self):
        return self.bt.get_profit()

    def _step(self, action):
        self._take_action(action)
        reward = self._get_reward()
        ob = self._get_observation()
        episode_over = self.ds.is_end()
        return ob, reward, episode_over, {}

    def _take_action(self, action):
        if ACTION_LOOKUP[action] == 'buy' and not self.bt.position:
            self.bt.buy(self.ds[0].close, self.balance)
        if ACTION_LOOKUP[action] == 'sell' and self.bt.position:
            self.bt.sell(self.ds[0].close)
        if not self.bt.ds.is_end():
            self.bt.go()

    def _reset(self):
        df = pd.read_csv('btc_etc.csv').rename(columns={
            'Close': 'close',
            'Date time': 'datetime',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Volume': 'volume'
        })
        self.balance = 1000.0
        self.ds = DataSeries(df, index=3)
        self.bt = BT(self.ds, self.balance)
        return self._get_observation()

    def _get_observation(self):
        keys = ['open', 'high', 'low', 'close']
        ob = []
        for n in range(4):
            for k in keys:
                ob.append(getattr(self.ds[n-3], k))
        ob.append(1.0 if self.bt.position is None else 0.0)
        return ob

    def _render(self, mode='human', close=False):
        pass
