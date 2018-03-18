import gym
import random
import numpy as np
import tensorflow as tf
from tflearn import conv_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from statistics import mean, median, mode
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
        
        self.operations = self.dqnutils.build_graph()
        
        session.run(tf.global_variables_initializer())

    def train(self):
        replay_memory = namedtuple("replay_memory", ["state", "action", "reward", "next_state", "done"]) # D
        replay_memory.state = np.zeros((self.cfg.replay_memory_size, self.cfg.agent_history_length, self.cfg.input_imgy, self.cfg.input_imgx))
        replay_memory.action = np.zeros((self.cfg.replay_memory_size))
        replay_memory.reward = np.zeros((self.cfg.replay_memory_size))
        replay_memory.next_state = np.zeros((self.cfg.replay_memory_size, self.cfg.agent_history_length, self.cfg.input_imgy, self.cfg.input_imgx))
        replay_memory.done = np.zeros((self.cfg.replay_memory_size))
        replay_memory_idx = 0

        total_steps = 0
        train_steps = 0
        replay_memory_initialized = False
        scores = []

        action_stat = {action_id:0 for action_id in range(self.wrapped_env.action_space_size)}

        self.session.run(self.operations["clone_Q_to_Q_target"])

        for episode in range(self.cfg.episodes):

            state = self.wrapped_env.get_initial_state()

            score = 0

            for _ in range(self.cfg.time_steps):
                Q_values_for_actions = self.operations["Q"].eval(session=self.session, feed_dict={self.operations["Q_state"]:[state]})
                print(np.round(Q_values_for_actions,2))

                action = self.wrapped_env.get_action(Q_values_for_actions)

                action_stat[action] += 1

                next_state, reward, done = self.wrapped_env.step(action)

                score+=reward

                replay_memory.state[replay_memory_idx] = state
                replay_memory.action[replay_memory_idx] = action
                replay_memory.reward[replay_memory_idx] = reward
                replay_memory.next_state[replay_memory_idx] = next_state
                replay_memory.done[replay_memory_idx] = done

                state = next_state

                if total_steps%self.cfg.target_network_update_frequency==0:
                    print("Cloning Q->Q_target")
                    self.session.run(self.operations["clone_Q_to_Q_target"])
                
                if (total_steps%self.cfg.update_frequency==0 or done) and not (replay_memory_idx<self.cfg.minibatch_size and not replay_memory_initialized):
                    if replay_memory_initialized:
                        batch_indexes = np.random.randint(self.cfg.replay_memory_size, size=self.cfg.minibatch_size)
                    else:
                        batch_indexes = np.random.randint(replay_memory_idx, size=self.cfg.minibatch_size)
                    
                    train_steps+=1
                    _, Q, action, y, b, cost = self.session.run(
                        [
                            self.operations["update"],
                            self.operations["Q"],
                            self.operations["action"],
                            self.operations["y"],
                            self.operations["b"],
                            self.operations["cost"]
                        ], feed_dict={
                        self.operations["Q_state"]: replay_memory.state[batch_indexes],
                        self.operations["Q_target_state"]: replay_memory.next_state[batch_indexes],
                        self.operations["action"]: replay_memory.action[batch_indexes],
                        self.operations["reward"]: replay_memory.reward[batch_indexes],
                        self.operations["done"]: replay_memory.done[batch_indexes]
                    })

                    # print("Q")
                    # print(np.round(Q,4))
                    # print("action")
                    # print(action)
                    # print("y")
                    # print(np.round(y, 4))
                    # print("b")
                    # print(np.round(b, 4))

                if replay_memory_idx==self.cfg.replay_memory_size-1:
                    replay_memory_idx = 0
                    replay_memory_initialized = True
                else:
                    replay_memory_idx += 1

                total_steps += 1

                if done:
                    break
            
            scores.append(score)

            print(f"Episode: {episode}")
            print(f"Score mean: {round(mean(scores),4)}")
            print(f"Score median: {median(scores)}")
            print(f"Action stat: {action_stat}")
            print(f"Train steps: {train_steps}")
            print(f"Epsilon: {round(self.cfg.epsilon,4)}")
            #print(f"Score mode: {mode(scores)}")
            print()

        self.saver.save(self.session, self.cfg.model_path)

    def evaluation(self):
        self.saver.restore(self.session, self.cfg.model_path)

        for _ in range(self.cfg.episodes):
            done = False
            rewards = []
            state = self.wrapped_env.get_initial_state()
            while not done:
                self.wrapped_env.render()
                Q_values_for_actions = self.operations["Q"].eval(session=self.session, feed_dict={self.operations["Q_state"]: [state]})
                action = np.argmax(Q_values_for_actions)
                state, reward, done = self.wrapped_env.step(action)
                rewards.append(reward)

            self.wrapped_env.close()