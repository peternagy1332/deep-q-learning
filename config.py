import os

class Config(object):
    def __init__(self):
        self.game_id = 'CartPole-v0'
        self.cropx = 84
        self.cropy = 84
        
        self.minibatch_size = 32
        self.replay_memory_size = 1000000 # N
        self.agent_history_length = 4 # Q network input
        self.target_network_update_frequency = 10000 # C
        self.discount_factor = 0.99
        self.action_repeat = 4 # every 4th frame
        self.update_frequency = 4
        
        # RMSProp
        self.learning_rate = 0.00025
        self.gradient_momentum = 0.95
        self.squared_gradient_momentum = 0.95
        self.min_squared_gradient = 0.01
        
        # epsilon-greedy
        self.initial_exploration = 1
        self.final_exploration = 0.1
        self.final_exploration_frame = 1000000
        self.epsilon = self.initial_exploration
        
        self.replay_start_size = 50000
        self.noop_max = 30
        
        # Arbitrary
        self.episodes = 20 # M
        self.time_steps = 200
        
        # Non-paper variables
        self.random_seed = 666
        self.epochs = 5
        self.snapshot_step = 1000
        self.model_path = os.path.join('Q')
