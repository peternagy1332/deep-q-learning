import tensorflow as tf
from tflearn import conv_2d, DNN
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import numpy as np

class DQNUtils(object):
    def __init__(self, cfg):
        self.cfg = cfg
       
    def __spawn_network(self):
        state = tf.placeholder(tf.uint8, [None, self.cfg.agent_history_length, self.cfg.cropy, self.cfg.cropx])
        net = tf.transpose(state, [0,2,3,1])
        net = conv_2d(net, 32, 8, strides=4, activation='relu')
        net = conv_2d(net, 64, 4, strides=2, activation='relu')
        net = conv_2d(net, 64, 3, strides=1, activation='relu')
        net = fully_connected(net, 512, activation='relu')
        net = fully_connected(net, self.cfg.wrapped_env.action_space.n)
        return state, net

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

        actions = tf.placeholder(tf.uint8, [None])
        y = tf.placeholder(tf.uint8, [None])

        actions_onehot = tf.one_hot(actions, self.cfg.wrapped_env.action_space_size, dtype=tf.uint8)

        Q_values_for_actions = tf.reduce_sum(tf.multiply(Q, actions_onehot), reduction_indices=1)
        
        cost = tf.squared_difference(Q_values_for_actions, y)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.cfg.learning_rate,momentum=self.cfg.gradient_momentum)

        # The update step operation
        update = optimizer.minimize(cost, var_list=Q_network_params)

        return Q_state, Q, Q_target_state, Q_target, clone_Q_to_Q_target, actions, y, update

    
    def __train_data_preprocessing(self, replay_memory):
        random_batch_indexes = np.random.randint(self.cfg.replay_memory_size, size=self.cfg.minibatch_size)
        prevous_state = replay_memory.prevous_state[random_batch_indexes]
        action = replay_memory.action[random_batch_indexes]
        reward = replay_memory.reward[random_batch_indexes]
        next_state = replay_memory.next_state[random_batch_indexes]
        
        return X, Y
