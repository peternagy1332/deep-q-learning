import os


class Config(object):
    def __init__(self):
        self.game_id = 'CartPole-v0'
        self.input_imgx = 84
        self.input_imgy = 84
        self.cropx = 250
        self.cropy = 250

        self.minibatch_size = 32
        self.replay_memory_size = 10000  # N
        self.agent_history_length = 4  # Q network input
        self.target_network_update_frequency = 10000  # C
        self.discount_factor = 0.99
        self.update_frequency = 4

        # Neural network parameters
        self.learning_rate = 0.0001

        # epsilon-greedy
        self.initial_exploration = 1
        self.final_exploration = 0.1
        self.final_exploration_frame = 10000
        self.epsilon = self.initial_exploration
        self.epsilon_annealer = (self.initial_exploration - self.final_exploration) / self.final_exploration_frame

        self.replay_start_size = 50000

        # Arbitrary
        self.episodes = 100000  # M
        self.time_steps = 200

        # Non-paper variables
        self.random_seed = 1332
        self.model_path = os.path.join('Q')
