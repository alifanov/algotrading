import gym
from gym import error, spaces, utils
from gym.utils import seeding

from mikasa import *

ACTION_LOOKUP = {
    0: 'nop',
    1: 'buy',
    2: 'sell'
}


class MikasaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Box(low=0, high=2.0, shape=(1,))
        self.observation_space = spaces.Box(low=0, high=1000.0, shape=(6, ))

    def _get_reward(self):
        return self.bt.get_profit()

    def _step(self, action):
        self._take_action(action)
        reward = self._get_reward()
        ob = self._get_observation()
        episode_over = self.ds.is_end()
        return ob, reward, episode_over, {}

    def _take_action(self, action):
        if ACTION_LOOKUP[action] == 1:
            self.bt.buy(self.ds[0].close, self.balance)
        if ACTION_LOOKUP[action] == 2:
            self.bt.sell(self.ds[0].close)

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
        self.ds = DataSeries(df)
        self.bt = BT(self.ds, self.balance)
        return self._get_observation()

    def _get_observation(self):
        d = self.ds[0]
        return (d.open, d.high, d.low, d.close, d.volume, 1 if self.bt.position is None else 0)

    def _render(self, mode='human', close=False):
        pass
