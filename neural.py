import tensorflow as tf
import datetime
import time
import os
from tflearn import conv_2d, mean_square
from tflearn.layers.core import fully_connected


class DQNUtils(object):
    """All neural network utilities. Network building, loss calculation, update operation."""
    def __init__(self, cfg, wrapped_env):
        self.cfg = cfg
        self.wrapped_env = wrapped_env

    def generate_model_name(self):
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d--%H:%M:%S')
        return os.path.join("models", f"model--{st}", "q-model")

    def __spawn_network(self):
        input_state_placeholder = tf.placeholder(tf.uint8, [None, self.cfg.agent_history_length, self.cfg.input_imgy, self.cfg.input_imgx])
        input_state = tf.transpose(input_state_placeholder, [0, 2, 3, 1])
        input_state = tf.cast(input_state, tf.float32)
        net = conv_2d(input_state, 32, 8, strides=4, activation='relu')
        net = conv_2d(net, 64, 4, strides=2, activation='relu')
        net = conv_2d(net, 64, 3, strides=1, activation='relu')
        net = fully_connected(net, 512, activation='relu')
        Q_values_for_actions = fully_connected(net, self.wrapped_env.action_space_size)
        return input_state_placeholder, Q_values_for_actions

    def build_graph(self):
        # Spawn Q
        Q_state, Q = self.__spawn_network()
        Q_network_params = tf.trainable_variables()
        Q_values_for_actions = Q

        # Spawn Q_target (the frozen one)
        Q_target_state, Q_target = self.__spawn_network()
        Q_target_network_params = tf.trainable_variables()[len(Q_network_params):]
        Q_target_values_for_actions = Q_target

        # Operation to freeze Q and save it to Q_target
        clone_Q_to_Q_target = [Q_target_network_params[i].assign(Q_network_params[i]) for i in range(len(Q_target_network_params))]

        # Placeholders
        action = tf.placeholder(tf.uint8, [None])
        reward = tf.placeholder(tf.uint8, [None])
        done = tf.placeholder(tf.uint8, [None])

        # Preprocessing
        undone = tf.ones((self.cfg.minibatch_size), dtype=tf.float32) - tf.cast(done, tf.float32)
        reward_clip = tf.cast(tf.clip_by_value(tf.cast(reward, tf.int32), -1, 1), tf.float32)
        action_onehot = tf.one_hot(action, depth=self.wrapped_env.action_space_size, dtype=tf.float32)

        # Desired rewards are calculated using the Bellman equation
        y = reward_clip + undone * tf.to_float(self.cfg.discount_factor) * tf.reduce_max(Q_target_values_for_actions, reduction_indices=1)

        action_q_values = tf.reduce_max(tf.multiply(Q_values_for_actions, action_onehot))

        # Loss calculation and optimization
        b = tf.multiply(action_onehot, y[:, tf.newaxis]) + tf.one_hot(action, depth=self.wrapped_env.action_space_size, on_value=0.0, off_value=1.0)*Q_values_for_actions

        cost = mean_square(Q_values_for_actions, b)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.cfg.learning_rate)

        update = optimizer.minimize(cost, var_list=Q_network_params)

        return {
            "Q_state": Q_state,
            "Q": Q,
            "Q_target_state": Q_target_state,
            "Q_taregt": Q_target,
            "clone_Q_to_Q_target": clone_Q_to_Q_target,
            "update": update,
            "action": action,
            "reward": reward,
            "done": done,
            "action_q_values": action_q_values
        }

    def clone_Q_to_Q_target(self, session, clone_Q_to_Q_target):
        """Another, frozen Q network (Q target) increases training efficiency."""
        print("Cloning Q weights to Q target network.")
        session.run(clone_Q_to_Q_target)
