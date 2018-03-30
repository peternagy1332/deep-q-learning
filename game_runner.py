import numpy as np
import tensorflow as tf
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from statistics import mean, median
from neural import DQNUtils
from config import Config
from environment_wrapper import EnvironmentWrapper
from replay_memory import ReplayMemory


class GameRunner(object):
    """Coordinates the training and evaluation of each network."""
    def __init__(self, session, default_config, model_dir):
        self.session = session
        self.saver = tf.train.Saver()

        self.cfg = Config(default_config, model_dir)

        self.wrapped_env = EnvironmentWrapper(self.cfg)
        self.dqnutils = DQNUtils(self.cfg, self.wrapped_env)

        self.operations = self.dqnutils.build_graph()

        # Train episode id <-> scores list
        self.eval_stats = self.cfg.get_eval_stats()

        self.initialize_model()

    def initialize_model(self):
        self.session.run(tf.global_variables_initializer())

        if os.path.exists(os.path.join(self.cfg.model_dir, "q-model.meta")):
            print("Loading existing model from: ", self.cfg.model_dir)

            latest_checkpoint = tf.train.latest_checkpoint(self.cfg.model_dir)
            self.saver.restore(self.session, latest_checkpoint)
        else:
            print("Model not found: ", self.cfg.model_dir, ". Creating new model with given name.")

    def train(self):
        """Trains the Q network using the replay memory and the frozen Q
        target network. Afterwards it saves the Q model."""
        print("Starting training of model: ", self.cfg.model_dir)

        replay_memory = ReplayMemory(self.cfg)

        scores = self.cfg.get_scores_list()

        try:

            for self.cfg.episode_counter in range(self.cfg.episode_counter, self.cfg.episodes):

                state = self.wrapped_env.get_initial_state()
                self.cfg.set_action_stat(self.wrapped_env.action_space_size)

                score = 0

                for _ in range(self.cfg.time_steps):
                    Q_values_for_actions = self.operations["Q"].eval(
                        session=self.session,
                        feed_dict={
                            self.operations["Q_state"]: [state]
                        }
                    )

                    action = self.wrapped_env.get_action(Q_values_for_actions)

                    next_state, reward, done = self.wrapped_env.step(action)

                    replay_memory.record_experience(state, action, reward, next_state, done)

                    state = next_state

                    # Clone Q network weights every self.cfg.target_network_update_frequency step.
                    if self.cfg.total_steps_counter % self.cfg.target_network_update_frequency == 0:
                        self.session.run(self.operations["clone_Q_to_Q_target"])

                    # Escalate a gradient update step on Q network every self.cfg.update_frequency step (if we have enough data).
                    if (self.cfg.total_steps_counter % self.cfg.update_frequency == 0 or done) and self.cfg.total_steps_counter >= self.cfg.replay_start_size:

                        replay_memory_sample = replay_memory.sample()

                        self.session.run(self.operations["update"], feed_dict={
                            self.operations["Q_state"]: replay_memory_sample.state,
                            self.operations["Q_target_state"]: replay_memory_sample.next_state,
                            self.operations["action"]: replay_memory_sample.action,
                            self.operations["reward"]: replay_memory_sample.reward,
                            self.operations["done"]: replay_memory_sample.done
                        })

                        self.cfg.train_steps_counter += 1

                    self.cfg.action_stat[action] += 1
                    score += reward
                    self.cfg.total_steps_counter += 1

                    if done:
                        break

                scores.append(score)

                print("Episode: ", self.cfg.episode_counter)
                print("\tScore mean: ", round(mean(scores),2))
                print("\tScore median: ", median(scores))
                print("\tScore mode: ", max(set(scores), key=scores.count))
                print("\tAction stat: ", self.cfg.action_stat)
                print("\tTotal steps: ", self.cfg.total_steps_counter)
                print("\tTrain steps: ", self.cfg.train_steps_counter)
                print("\tEpsilon: ", round(self.cfg.epsilon, 2))
                print()

                # Evaluating
                if (self.cfg.episode_counter+1) % 50 == 0:
                    self.evaluation(self.cfg.episode_counter)
                
        except KeyboardInterrupt:
            print("Saving model to ", self.cfg.model_dir)

            self.saver.save(self.session, os.path.join(self.cfg.model_dir, "q-model"))

            self.cfg.save(scores, self.eval_stats)

            self.evaluation(self.cfg.episode_counter)

            self.wrapped_env.close()

    def evaluation(self, train_episode_counter):
        """Evaluates a given Q model in a game environment."""

        print("Starting evaluation of model: ", self.cfg.model_dir)

        scores = []
        for episode in range(self.cfg.eval_episodes):
            done = False
            score = 0
            state = self.wrapped_env.get_initial_state()
            step = 0
            while not done and step < self.cfg.time_steps:

                Q_values_for_actions = self.operations["Q"].eval(
                    session=self.session,
                    feed_dict={
                        self.operations["Q_state"]: [state]
                    }
                )

                action = np.argmax(Q_values_for_actions)
                state, reward, done = self.wrapped_env.step(action)

                score += reward
                step += 1

            scores.append(score)

        self.eval_stats.setdefault(train_episode_counter, scores)

        # Draw stat
        fig = plt.figure()
        x = []
        y = []
        for episode, scores in self.eval_stats.items():
                x += [episode]*len(scores)
                y.extend(scores)
        plt.plot(x, y, ".")
        plt.xlabel("Train episodes")
        plt.ylabel("Evaluation scores")
        fig.savefig(os.path.join(self.cfg.model_dir, "eval"))
        plt.close(fig)
