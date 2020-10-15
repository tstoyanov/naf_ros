import numpy as np
import random

import utils
from replay_memory import *

class PrioritizedReplayMemory(ReplayMemory):
    def __init__(self, capacity, alpha):
        """Create Prioritized Replay Memory.
        Parameters
        ----------
        capacity: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        """
        super().__init__(capacity)
        assert alpha >= 0
        self.alpha = alpha

        it_capacity = 1
        while it_capacity < capacity:
            it_capacity *= 2

        self.it_sum = utils.SumSegmentTree(it_capacity)
        self.it_min = utils.MinSegmentTree(it_capacity)
        self.max_priority = 1.0
        
    def push(self, *args):
        """See ReplayMemory"""
        idx = self.position
        super().push(*args)
        self.it_sum[idx] = self.max_priority ** self.alpha
        self.it_min[idx] = self.max_priority ** self.alpha
        
    def sample_proportional(self, batch_size):
        res = []
        p_total = self.it_sum.sum(0, len(self.memory) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self.it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.
        compared to ReplayMemory.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        transitions
        """
        assert beta > 0

        idxes = self.sample_proportional(batch_size)

        weights = []
        p_min = self.it_min.min() / self.it_sum.sum()
        max_weight = (p_min * len(self.memory)) ** (-beta)

        for idx in idxes:
            p_sample = self.it_sum[idx] / self.it_sum.sum()
            weight = (p_sample * len(self.memory)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        
        transitions = [self.memory[i] for i in idxes]
        return transitions

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.memory)
            self.it_sum[idx] = priority ** self.alpha
            self.it_min[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
            