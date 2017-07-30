import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents.sarsa import SarsaAgent
from rl.policy import BoltzmannQPolicy

from poloniex.gym_mikasa import MikasaLast4Env

# create Mikasa gym env
env = MikasaLast4Env()
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

print('Actions: ', nb_actions)
print('Observations: ', (1,) + env.observation_space.shape)

# create model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

# configure agent
policy = BoltzmannQPolicy()
dqn = SarsaAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=10, policy=policy)
dqn.compile(Adam(lr=1e-5), metrics=['mae'])

# run agent
history = dqn.fit(env, nb_steps=10000, visualize=False, verbose=1, log_interval=100)
plt.plot(history.history['episode_reward'])
plt.show()
