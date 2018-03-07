import gym
import random
import numpy as np
import tensorflow as tf
from tflearn import conv_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from neural import DQNUtils
from config import Config
from environment_wrapper import EnvironmentWrapper

class GameRunner(object):
    def __init__(self, session):
        self.session = session
        self.saver = tf.train.Saver(max_to_keep=5)
        self.cfg = Config()
        self.dqnutils = DQNUtils(self.cfg)
        self.wrapped_env = EnvironmentWrapper(self.cfg)
        self.Q_state, self.Q, self.Q_target_state, self.Q_target, self.clone_Q_to_Q_target, self.actions, self.y, self.update = self.dqnutils.build_graph()

    def train(self):
        replay_memory = namedtuple("replay_memory", ["state", "action", "reward", "next_state", "y"]) # D
        replay_memory.state = np.zeros((self.cfg.replay_memory_size, self.cfg.agent_history_length, self.cfg.cropy, self.cfg.cropx))
        replay_memory.action = np.zeros((self.cfg.replay_memory_size))
        replay_memory.reward = np.zeros((self.cfg.replay_memory_size))
        replay_memory.next_state = np.zeros((self.cfg.replay_memory_size, self.cfg.agent_history_length, self.cfg.cropy, self.cfg.cropx))
        replay_memory.y = np.zeros((self.cfg.replay_memory_size))
        replay_memory_idx = 0

        total_steps = 0

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

                Q_target_values_for_actions = self.Q_target.eval(session=self.session, feed_dict={self.Q_target_state:[next_state]})

                clipped_reward = np.clip(reward, -1, 1)
                if done:
                    replay_memory.y[replay_memory_idx] = clipped_reward
                else:
                    replay_memory.y[replay_memory_idx] = clipped_reward + self.cfg.discount_factor*np.max(Q_target_values_for_actions)

                state = next_state
                rewards.append(reward)

                if total_steps%self.cfg.target_network_update_frequency==0:
                    self.session.run(self.clone_Q_to_Q_target)

                if total_steps%self.cfg.update_frequency==0 or done:
                    batch_indexes = np.random.randint(self.cfg.replay_memory_size, size=self.cfg.minibatch_size)
                    actions = replay_memory.action[batch_indexes]
                    y = replay_memory.y[batch_indexes]
                    self.session.run(self.update, feed_dict={self.actions: actions, self.y: y})

                if replay_memory_idx%self.cfg.replay_memory_size==0:
                    replay_memory_idx=0
                else:
                    replay_memory_idx+=1

                total_steps+=1

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