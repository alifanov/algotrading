import numpy as np
import time
import gym
env = gym.make('CartPole-v0')

from es import EvolutionStrategy
from keras.models import Sequential
from keras.layers import Dense


def get_model():
    model = Sequential()
    model.add(Dense(16, input_dim=4, activation='relu'))
    # model.add(Dense(256, activation='relu'))
    model.add(Dense(2, activation='relu'))

    model.compile(optimizer='Adam', loss='mse')
    return model


def get_reward(weights):
    model = get_model()
    model.set_weights(weights)
    total_steps = 0
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            # env.render()
            action = np.argmax(model.predict(np.expand_dims(observation, 0)))
            observation, reward, done, info = env.step(action)
            total_steps += 1
            if done:
                break
    reward = total_steps / 20.0
    return reward - 100.0

model = get_model()

es = EvolutionStrategy(model.get_weights(), get_reward, population_size=50, sigma=0.1, learning_rate=0.001)
es.run(1000, print_step=1)

