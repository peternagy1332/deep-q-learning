import gym
import random
import numpy as np
import tensorflow as tf
from tflearn import conv_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from neural import DQNUtils
from config import Config
from environment_wrapper import EnvironmentWrapper
from collections import namedtuple

class GameRunner(object):
    def __init__(self, session):
        self.session = session
        self.saver = tf.train.Saver(max_to_keep=5)
        self.cfg = Config()
        self.wrapped_env = EnvironmentWrapper(self.cfg)
        self.dqnutils = DQNUtils(self.cfg, self.wrapped_env)
        self.Q_state, self.Q, self.Q_target_state, self.Q_target, self.clone_Q_to_Q_target, self.update, self.actions, self.rewards, self.dones = self.dqnutils.build_graph()
        session.run(tf.global_variables_initializer())

    def train(self):
        replay_memory = namedtuple("replay_memory", ["state", "action", "reward", "next_state", "done"]) # D
        replay_memory.state = np.zeros((self.cfg.replay_memory_size, self.cfg.agent_history_length, self.cfg.cropy, self.cfg.cropx))
        replay_memory.action = np.zeros((self.cfg.replay_memory_size))
        replay_memory.reward = np.zeros((self.cfg.replay_memory_size))
        replay_memory.next_state = np.zeros((self.cfg.replay_memory_size, self.cfg.agent_history_length, self.cfg.cropy, self.cfg.cropx))
        replay_memory.done = np.zeros((self.cfg.replay_memory_size))
        replay_memory_idx = 0

        total_steps = 0
        replay_memory_initialized = False

        for _ in range(self.cfg.episodes):
            state = self.wrapped_env.get_initial_state()

            rewards = []

            for _ in range(self.cfg.time_steps):
                Q_values_for_actions = self.Q.eval(session=self.session, feed_dict={self.Q_state:[state]})

                action = self.wrapped_env.get_action(Q_values_for_actions)

                next_state, reward, done = self.wrapped_env.step(action)

                replay_memory.state[replay_memory_idx] = state
                replay_memory.action[replay_memory_idx] = action
                replay_memory.reward[replay_memory_idx] = reward
                replay_memory.next_state[replay_memory_idx] = next_state
                replay_memory.done[replay_memory_idx] = done

                state = next_state
                rewards.append(reward)

                if total_steps%self.cfg.target_network_update_frequency==0:
                    print("Cloning Q->Q_target")
                    self.session.run(self.clone_Q_to_Q_target)

                if (total_steps%self.cfg.update_frequency==0 or done) and not (replay_memory_idx<self.cfg.minibatch_size and not replay_memory_initialized):
                    if replay_memory_initialized:
                        batch_indexes = np.random.randint(self.cfg.replay_memory_size, size=self.cfg.minibatch_size)
                    else:
                        batch_indexes = np.random.randint(replay_memory_idx, size=self.cfg.minibatch_size)
                    print("Train!")
                    self.session.run(self.update, feed_dict={
                        self.Q_state: replay_memory.state[batch_indexes],
                        self.Q_target_state: replay_memory.next_state[batch_indexes],
                        self.actions: replay_memory.action[batch_indexes],
                        self.rewards: replay_memory.reward[batch_indexes],
                        self.dones: replay_memory.done[batch_indexes]
                    })

                if replay_memory_idx%self.cfg.replay_memory_size==0:
                    replay_memory_idx=0
                    replay_memory_initialized = True
                else:
                    replay_memory_idx+=1

                total_steps+=1

                if done:
                    break

        self.saver.save(self.session, self.cfg.model_path)

    def evaluation(self):
        self.saver.restore(self.session, self.cfg.model_path)

        for _ in range(self.cfg.episodes):
            done = False
            rewards = []
            state = self.wrapped_env.get_initial_state()
            while not done:
                self.wrapped_env.render()
                Q_values_for_actions = self.Q.eval(session=self.session, feed_dict={self.Q_state: [state]})
                action = np.argmax(Q_values_for_actions)
                state, reward, done = self.wrapped_env.step(action)
                rewards.append(reward)

            self.wrapped_env.close()