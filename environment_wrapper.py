import random
import numpy as np
import gym

class EnvironmentWrapper(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.env = gym.make(self.cfg.game_id)
        self.action_space_size = self.env.action_space.n
        self.state_buffer = np.zeros((self.cfg.action_repeat-1, self.cfg.cropy, self.cfg.cropx))
    
    def get_initial_state(self):
        initial_frame = self.__preprocess_frame(self.env.reset(mode='rgb_array'))
        initial_state = np.stack([initial_frame for _ in range(self.cfg.action_repeat)])
        self.state_buffer = initial_state[:self.cfg.action_repeat-1]
        return initial_state

    def __preprocess_frame(self, frame):
        """
        frame.shape = (400, 600, 3)
        output.shape = (84, 84) # grayscale
        """
        r, g, b = frame[:,:,0], frame[:,:,1], frame[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        h, w = gray.shape
        starty = h//2 - self.cfg.cropy//2
        startx = w//2 - self.cfg.cropx//2

        return gray[starty:starty+self.cfg.cropy, startx:startx+self.cfg.cropx].astype(np.uint8)


    def step(self, action):
        _, reward, done, _ = self.env.step(action)
        
        preprocessed_frame = self.__preprocess_frame(self.env.render(mode='rgb_array'))        

        next_state = np.zeros((self.cfg.action_repeat, self.cfg.cropy, self.cfg.cropx))

        next_state[:self.cfg.action_repeat-1] = self.state_buffer
        next_state[self.cfg.action_repeat] = preprocessed_frame
 
        return next_state, reward, done
        
    def get_action(self, Q_values_for_actions):

        # epsilon-greedy
        if random.random() <= self.cfg.epsilon:
            action = random.randrange(self.env.action_space.n)
        else:
            action = np.argmax(Q_values_for_actions)
        
        # anneal epsilon
        if self.cfg.epsilon>self.cfg.final_exploration:
            self.cfg.epsilon = (self.cfg.initial_exploration - self.cfg.final_exploration) / self.cfg.final_exploration_frame
        
        return action

    def close(self):
        self.env.close()