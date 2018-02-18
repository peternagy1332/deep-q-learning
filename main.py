import gym
import random
import numpy as np
from statistics import mean, median
from collections import Counter

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

env = gym.make('CartPole-v0')
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 50
discount_factor = 0.95

def game(Q):
    training_data = []
    scores = []

    for _ in range(initial_games):

        env.reset()
        score = 0
        game_memory = []
        prev_observation = None

        for _ in range(goal_steps):


            if prev_observation is not None:
                prev_qualities = Q.predict(np.matrix(prev_observation))[0]

                action = np.argmax(prev_qualities)
                
                observation, reward, done, _ = env.step(action)

                prev_qualities[action] = reward+discount_factor*max(Q.predict(np.matrix(observation))[0])

                game_memory.append((prev_observation, prev_qualities))

            else:
                action = env.action_space.sample()

                observation, reward, done, _ = env.step(action)
            
            prev_observation = observation

            score += reward
            scores.append(score)

            #env.render()

            training_data.extend(game_memory)

            if done:
                break

    print(Counter(scores))
    print('max',max(scores))
    print('min',min(scores))
    print('mean',mean(scores))
    print('median',median(scores))

    return training_data


def neural_network_model():
    network = input_data(shape=[None, len(env.observation_space.high)], name='input')
    network = fully_connected(network, 10, activation='relu')
    network = dropout(network, 0.8)
    network = fully_connected(network, 10, activation='relu')
    network = dropout(network, 0.8)
    network = fully_connected(network, env.action_space.n, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=1e-3, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network)
    return model


def train_model(training_data, model):
    X = np.array([i[0] for i in training_data])
    Y = np.array([i[1] for i in training_data])
    model.fit(X, Y, n_epoch=5, snapshot_step=1000, show_metric=True, run_id='openaistuff')
    return model


Q = neural_network_model()

for i in range(100):
    print('Population:',i)
    training_data = game(Q)
    Q = train_model(training_data, Q)

