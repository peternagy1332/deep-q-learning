import tensorflow as tf
from tflearn import conv_2d, DNN
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import numpy as np

class DQNUtils(object):
    def __init__(self, cfg, wrapped_env):
        self.cfg = cfg
        self.wrapped_env = wrapped_env
       
    def __spawn_network(self):
        input_state_placeholder = tf.placeholder(tf.uint8, [None, self.cfg.agent_history_length, self.cfg.cropy, self.cfg.cropx])
        input_state = tf.cast(input_state_placeholder, tf.float32)
        input_state = tf.transpose(input_state, [0,2,3,1])
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

        # Spawn Q_target (the frozen one)
        Q_target_state, Q_target = self.__spawn_network()
        all_network_params = tf.trainable_variables()
        Q_target_network_params = all_network_params[len(Q_network_params):]

        # Operation to freeze Q and save it to Q_target
        clone_Q_to_Q_target = [Q_target_network_params[i].assign(Q_network_params[i]) for i in range(len(Q_target_network_params))]

        # Placeholders
        actions = tf.placeholder(tf.uint8, [None])
        rewards = tf.placeholder(tf.uint8, [None])
        dones = tf.placeholder(tf.uint8, [None])

        # Preprocessing
        dones2 = tf.cast(tf.bitwise.invert(dones), tf.float32)
        rewards2 = tf.cast(tf.clip_by_value(tf.cast(rewards, tf.int32), -1, 1), tf.float32)

        # Q values
        a = dones2*tf.to_float(self.cfg.discount_factor)
        b = tf.reduce_max(Q_target, reduction_indices=1)
        y = rewards2 + tf.multiply(a,b)
        
        action_q_values = tf.reduce_max(Q, reduction_indices=1)

        # Loss optimization
        cost = tf.squared_difference(action_q_values, y)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.cfg.learning_rate,momentum=self.cfg.gradient_momentum)

        # The update step operation
        update = optimizer.minimize(cost, var_list=Q_network_params)

        return Q_state, Q, Q_target_state, Q_target, clone_Q_to_Q_target, update, actions, rewards, dones