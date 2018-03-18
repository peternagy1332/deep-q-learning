import os

class Config(object):
    def __init__(self):
        self.game_id = 'CartPole-v0'
        self.input_imgx=84
        self.input_imgy=84
        self.cropx = 250
        self.cropy = 250
        
        self.minibatch_size = 32
        self.replay_memory_size = 10000 # N
        self.agent_history_length = 4 # Q network input
        self.target_network_update_frequency = 10000 # C
        self.discount_factor = 0.99
        self.action_repeat = 4 # every 4th frame
        self.update_frequency = 4
        
        # RMSProp
        self.learning_rate = 0.0001
        self.gradient_momentum = 0.95
        self.squared_gradient_momentum = 0.95
        self.min_squared_gradient = 0.01
        
        # epsilon-greedy
        self.initial_exploration = 1
        self.final_exploration = 0.1
        self.final_exploration_frame = 10000
        self.epsilon = self.initial_exploration
        
        self.replay_start_size = 50000
        self.noop_max = 30
        
        # Arbitrary
        self.episodes = 100000 # M
        self.time_steps = 200
        
        # Non-paper variables
        self.random_seed = 666
        self.model_path = os.path.join('Q')
