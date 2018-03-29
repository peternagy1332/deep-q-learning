import random
import numpy as np
import gym
from scipy.misc import imsave, imresize


class EnvironmentWrapper(object):
    """Hides frame preprocessing and epsilon-greedy stepping."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.env = gym.make(self.cfg.game_id)
        self.action_space_size = self.env.action_space.n
        self.state_buffer = np.zeros((self.cfg.agent_history_length-1, self.cfg.input_imgy, self.cfg.input_imgx))

    def get_initial_state(self):
        """The initial state is self.cfg.agent_history_length of 2D zero matrices."""
        self.env.reset()
        initial_state = np.zeros((self.cfg.agent_history_length, self.cfg.input_imgy, self.cfg.input_imgx))
        self.state_buffer = initial_state[:self.cfg.agent_history_length-1]
        return initial_state

    def __preprocess_frame(self, frame):
        """
        frame.shape = (x, y, chanels) -> output.shape = (input_imgy, input_imgx) # grayscale
        """

        # Grayscale using luminance
        r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        # Centered cropping
        h, w = gray.shape
        starty = h//2 - self.cfg.cropy//2
        startx = w//2 - self.cfg.cropx//2

        cropped = gray[starty:starty+self.cfg.cropy, startx:startx+self.cfg.cropx]

        # Resizing grayscaled, cropped image
        resized = imresize(cropped, (self.cfg.input_imgy, self.cfg.input_imgx), interp='bilinear', mode=None)

        return resized

    def step(self, action):
        """Take an action, then preprocess the rendered frame."""
        if self.cfg.action_repeat is not None:
            repeat = self.cfg.action_repeat
            done = False
            while not done and repeat > 0:
                _, reward, done, _ = self.env.step(action)
                repeat -= 1
        else:
            _, reward, done, _ = self.env.step(action)

        original_frame = self.env.render(mode='rgb_array')
        preprocessed_frame = self.__preprocess_frame(original_frame)

        next_state = np.zeros((self.cfg.agent_history_length, self.cfg.input_imgy, self.cfg.input_imgx))

        # [...previous self.cfg.agent_history_length-1 frames..., latest frame]
        next_state[:-1] = self.state_buffer
        next_state[-1] = preprocessed_frame

        # Sampling and visualizing network input
        # if done:
        #     imsave('assets/'+self.cfg.game_id+'/original.png', original_frame)
        #     for i in range(next_state.shape[0]):
        #         imsave('assets/'+self.cfg.game_id+'/net-input-'+str(i)+'.png', next_state[i])

        # Pushing the freshly preprocessed frame into the FIFO-like buffer.
        self.state_buffer[:-1] = self.state_buffer[1:]
        self.state_buffer[-1] = preprocessed_frame

        return next_state, reward, done

    def get_action(self, Q_values_for_actions):
        """Returns a random action with self.cfg.epsilon probability,
        otherwise the most beneficial action in long term."""

        # Epsilon-greedy action choosing
        if random.random() <= self.cfg.epsilon:
            action = random.randrange(self.env.action_space.n)
        else:
            action = np.argmax(Q_values_for_actions)

        # Anneal epsilon
        if self.cfg.epsilon > self.cfg.final_exploration:
            self.cfg.epsilon -= self.cfg.epsilon_annealer

        return action

    def close(self):
        self.env.close()

    def render(self):
        self.env.render()
