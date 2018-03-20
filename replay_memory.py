from collections import namedtuple
import numpy as np

class ReplayMemory(object):
    def __init__(self, cfg):
        """Initialization of "the mammalian" replay memory."""
        self.cfg = cfg

        # D in paper
        self.Memory = namedtuple("Memory", ["state", "action", "reward", "next_state", "done"])

        self.replay_memory = self.Memory(
            state=np.zeros((self.cfg.replay_memory_size, self.cfg.agent_history_length, self.cfg.input_imgy, self.cfg.input_imgx)),
            action=np.zeros((self.cfg.replay_memory_size)),
            reward=np.zeros((self.cfg.replay_memory_size)),
            next_state=np.zeros((self.cfg.replay_memory_size, self.cfg.agent_history_length, self.cfg.input_imgy, self.cfg.input_imgx)),
            done=np.zeros((self.cfg.replay_memory_size))
        )

        self.replay_memory_idx = 0
        self.replay_memory_initialized = False

    def record_experience(self, state, action, reward, next_state, done):
        self.replay_memory.state[self.replay_memory_idx] = state
        self.replay_memory.action[self.replay_memory_idx] = action
        self.replay_memory.reward[self.replay_memory_idx] = reward
        self.replay_memory.next_state[self.replay_memory_idx] = next_state
        self.replay_memory.done[self.replay_memory_idx] = done

        # Overwriting old experiences.
        if self.replay_memory_idx == self.cfg.replay_memory_size-1:
            self.replay_memory_idx = 0
            self.replay_memory_initialized = True
        else:
            self.replay_memory_idx += 1

    def not_enough_data_for_one_minibatch(self):
        """Don't update the network if there is not enough data
        for at least one full minibatch."""
        return self.replay_memory_idx < self.cfg.minibatch_size and not self.replay_memory_initialized

    def sample(self):
        """Returns self.cfg.minibatch_size of random experiences."""

        # If the replay memory is not initialized fully, we can choose data only from the initialized part.
        if self.replay_memory_initialized:
            batch_indexes = np.random.randint(self.cfg.replay_memory_size, size=self.cfg.minibatch_size)
        else:
            batch_indexes = np.random.randint(self.replay_memory_idx, size=self.cfg.minibatch_size)

        return self.Memory(
            state=self.replay_memory.state[batch_indexes],
            action=self.replay_memory.action[batch_indexes],
            reward=self.replay_memory.reward[batch_indexes],
            next_state=self.replay_memory.next_state[batch_indexes],
            done=self.replay_memory.done[batch_indexes]
        )
