import random
import numpy as np
import gym 
from scipy.misc import imsave, imresize


class EnvironmentWrapper(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.env = gym.make(self.cfg.game_id)
        self.action_space_size = self.env.action_space.n
        self.state_buffer = np.zeros((self.cfg.action_repeat-1, self.cfg.input_imgy, self.cfg.input_imgx))
    
    def get_initial_state(self):
        self.env.reset()
        initial_state = np.zeros((self.cfg.action_repeat, self.cfg.input_imgy, self.cfg.input_imgx))
        self.state_buffer = initial_state[:self.cfg.action_repeat-1]
        return initial_state

    def __preprocess_frame(self, frame):
        """
        frame.shape = (400, 600, 3)
        output.shape = (84, 84) # grayscale
        """
        
        # Grayscale
        r, g, b = frame[:,:,0], frame[:,:,1], frame[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        # Crop
        h, w = gray.shape
        starty = h//2 - self.cfg.cropy//2
        startx = w//2 - self.cfg.cropx//2

        cropped = gray[starty:starty+self.cfg.cropy, startx:startx+self.cfg.cropx]

        # Resize
        shrinked = imresize(cropped, (84,84), interp='bilinear', mode=None)

        return shrinked


    def step(self, action):
        _, reward, done, _ = self.env.step(action)
        
        preprocessed_frame = self.__preprocess_frame(self.env.render(mode='rgb_array'))        

        next_state = np.zeros((self.cfg.action_repeat, self.cfg.input_imgy, self.cfg.input_imgx))
        
        next_state[:-1] = self.state_buffer
        next_state[-1] = preprocessed_frame

        # if done:
        #     for i in range(next_state.shape[0]):
        #         imsave(str(i)+'.png', next_state[i])

        self.state_buffer[:-1] = self.state_buffer[1:]
        self.state_buffer[-1] = preprocessed_frame

        return next_state, reward, done
        
    def get_action(self, Q_values_for_actions):

        # epsilon-greedy
        if random.random() <= self.cfg.epsilon:
            action = random.randrange(self.env.action_space.n)
        else:
            action = np.argmax(Q_values_for_actions)
        
        # anneal epsilon
        if self.cfg.epsilon>self.cfg.final_exploration:
            self.cfg.epsilon -= (self.cfg.initial_exploration - self.cfg.final_exploration) / self.cfg.final_exploration_frame
        
        return action

    def close(self):
        self.env.close()

    def render(self):
        self.env.render()