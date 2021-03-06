import numpy as np
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

from mikasa_gym import MikasaEnv

from matplotlib import pyplot as plt

# create Mikasa gym env
env = MikasaEnv(source_filename='btc_etc_first100.csv')
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# create model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

# configure agent
policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mse'])

# run agent
history = dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
plt.plot(history.history['episode_reward'])
plt.show()
