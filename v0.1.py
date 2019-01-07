# -*- coding: utf-8 -*-
import gym
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from cartpole import CartPoleEnv


class DQN:
    def __init__(self, hidden_1, hidden_2, env):
        self.env = env
        self.memory = []

        # define hyper parameters
        self.gamma = 0.95
        self.epsilon = 1.0
        self.decay = 0.995
        self.learning_rate = 0.001

        # define model parameters
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.hidden_1, input_dim=self.env.observation_space, activation=tf.nn.relu),
            tf.keras.layers.Dense(self.hidden_2, activation=tf.nn.relu),
            tf.keras.layers.Dense(self.env.action_space.n+3, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate),
                      loss='mse')
        return model

    def remember(self, state, reward, action, state_, done):
        self.memory.append((state, reward, action, state_, done))

    def replay(self):
        batch_size = 32                             # shuffle the data
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        # samples = self.memory[-32:]
        for state, reward, action, state_, done in samples:
            target = self.model.predict(state)
            if done:
                # in case of episode termination
                target[0][action] = reward
            else:
                # predict Q' and compute the target function (reward + gamma * Q'(s', a))
                Q_ = reward + self.gamma * max(self.model.predict(state_)[0])
                target[0][action] = reward + self.gamma * Q_
            self.model.fit(state, target, epochs=1, verbose=0)

    def act(self, state):
        if self.epsilon > 0.01:
            self.epsilon *= self.decay
        if np.random.random() <= self.epsilon:
            return self.env.action_space.sample(), 10
        else:
            return np.argmax(self.model.predict(state)[0][0:2]), np.argmax(self.model.predict(state)[0][2:5])

    def save_model(self, filename):
        self.model.save_weights(filename)

    def load_model(self, filename):
        self.model.load_weights(filename)


if __name__ == '__main__':
    env = CartPoleEnv()
    agent = DQN(24, 24, env)
    # agent.load_model('bot.h5')

    episode_data = []
    score_data = []
    episode_data_ = []
    score_data_ = []

    # Learning
    for episode in range(5000):
        state = env.reset()
        state = np.reshape(state, [1, 4])                           # reshape from [[a, b]] to [a, b]

        for t in range(1000):
            action, force = agent.act(state)

            # given Q values for each force and max arg of it, apply the corresponding force to the model
            if force == 2:
                env.force_mag = 6
            elif force == 3:
                env.force_mag = 8
            else:
                env.force_mag = 10

            # perform one step
            state_, reward, done, info = env.step(action)
            state_ = np.reshape(state_, [1, 4])

            # reward function
            reward = reward if not done else -20                # additional penalty for a loss
            reward -= 1.0 * abs(state_[0][0])                   # penalty for moving too far away, increasing linearly

            # archive the step and perform fitting to the model
            agent.remember(state, reward, action, state_, done)
            # agent.replay()

            # state = next state
            state = state_

            if done:
                print('Episode {} ended after {} time steps, eps = {:.2}.'.format(episode, t, agent.epsilon))
                episode_data.append(episode)
                score_data.append(t)
                break
        agent.replay()
    # agent.save_model('bot.h5')

    # Performance
    for episode in range(30):
        state = env.reset()
        state = np.reshape(state, [1, 4])
        agent.epsilon = 0
        for t in range(1000):
            env.render()
            action, force = agent.act(state)
            if force == 2:
                env.force_mag = 6
            elif force == 3:
                env.force_mag = 8
            else:
                env.force_mag = 10
            state_, reward, done, info = env.step(action)

            state_ = np.reshape(state_, [1,4])
            state = state_
            if done or t == 1000:
                episode_data_.append(episode)
                score_data_.append(t)
                print('Trial {} ended with score {}'.format(episode, t))
                break

    plt.plot(episode_data, score_data)
    plt.xlabel('Episode number')
    plt.ylabel('Duration [time steps]')
    plt.title('Learning')

    plt.show()

    plt.plot(episode_data_, score_data_)
    plt.xlabel('Episode number')
    plt.ylabel('Duration [time steps]')
    plt.title('Performance')

    plt.show()
