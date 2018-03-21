from statistics import mean, median
import numpy as np
import tensorflow as tf
import os
from neural import DQNUtils
from config import Config
from environment_wrapper import EnvironmentWrapper
from replay_memory import ReplayMemory


class GameRunner(object):
    """Coordinates the training and evaluation of each network."""
    def __init__(self, session, model_dir):
        self.session = session
        self.saver = tf.train.Saver()
        self.model_dir = model_dir
        self.cfg = Config()
        self.wrapped_env = EnvironmentWrapper(self.cfg)
        self.dqnutils = DQNUtils(self.cfg, self.wrapped_env)
        self.operations = self.dqnutils.build_graph()
        self.initialize_model()
        
    def initialize_model(self):
        self.session.run(tf.global_variables_initializer())

        if self.model_dir is not None:

            if os.path.exists(self.model_dir) and os.path.isdir(self.model_dir):
                print(f"Loading existing model from: {self.model_dir}")
                
                latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)
                self.saver.restore(self.session, latest_checkpoint)
            else:
                print(f"Model not found: {self.model_dir}. Creating new model with given name.")

        else:
            self.model_dir = self.dqnutils.generate_model_name()
            print(f"Creating new model: {self.model_dir}")

    def train(self):
        """Trains the Q network using the replay memory and the frozen Q
        target network. Afterwards it saves the Q model."""
        print(f"Starting training of model: {self.model_dir}")

        replay_memory = ReplayMemory(self.cfg)

        total_steps = 0
        train_steps = 0
        scores = []

        action_stat = {action_id: 0 for action_id in range(self.wrapped_env.action_space_size)}

        try:

            for episode in range(self.cfg.episodes):

                state = self.wrapped_env.get_initial_state()

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
                    if total_steps % self.cfg.target_network_update_frequency == 0:
                        self.dqnutils.clone_Q_to_Q_target(self.session, self.operations["clone_Q_to_Q_target"])

                    # Escalate a gradient update step on Q network every self.cfg.update_frequency step (if we have enough data).
                    if (total_steps % self.cfg.update_frequency == 0 or done) and \
                            not replay_memory.not_enough_data_for_one_minibatch():

                        replay_memory_sample = replay_memory.sample()

                        self.session.run(self.operations["update"], feed_dict={
                            self.operations["Q_state"]: replay_memory_sample.state,
                            self.operations["Q_target_state"]: replay_memory_sample.next_state,
                            self.operations["action"]: replay_memory_sample.action,
                            self.operations["reward"]: replay_memory_sample.reward,
                            self.operations["done"]: replay_memory_sample.done
                        })

                        train_steps += 1

                    action_stat[action] += 1
                    score += reward
                    total_steps += 1

                    if done:
                        break

                scores.append(score)

                print(f"Episode: {episode}")
                print(f"\tScore mean: {round(mean(scores),2)}")
                print(f"\tScore median: {median(scores)}")
                print(f"\tScore mode: {max(set(scores), key=scores.count)}")
                print(f"\tAction stat: {action_stat}")
                print(f"\tTotal steps: {total_steps}")
                print(f"\tTrain steps: {train_steps}")
                print(f"\tEpsilon: {round(self.cfg.epsilon,2)}")
                print()
        except KeyboardInterrupt:
            print(f"Saving model to {self.model_dir}")
            if self.model_dir[-7:] != 'q-model':
                self.model_dir = os.path.join(self.model_dir, 'q-model')

            self.saver.save(self.session, self.model_dir)

    def evaluation(self):
        """Evaluates a given Q model in a game environment."""

        print(f"Starting evaluation of model: {self.model_dir}")

        for _ in range(self.cfg.episodes):
            done = False
            rewards = []
            state = self.wrapped_env.get_initial_state()
            while not done:
                self.wrapped_env.render()
                Q_values_for_actions = self.operations["Q"].eval(
                    session=self.session,
                    feed_dict={
                        self.operations["Q_state"]: [state]
                    }
                )
                action = np.argmax(Q_values_for_actions)
                state, reward, done = self.wrapped_env.step(action)
                rewards.append(reward)

            self.wrapped_env.close()
