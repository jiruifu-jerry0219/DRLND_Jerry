import gym
gym.logger.set_level(40) # suppress warnings (please remove if gives error)
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
%matplotlib inline

import torch
torch.manual_seed(0) # set random seed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from Policy import Policy

class Agent():
    def __init__(self, env, n_episodes = 1000, max_t = 1000, gamma = 1.0, print_every=100):
        self.env = env
        self.env.seed(0)
        self.episodes = n_episodes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_t = max_t
        self.gamma = gamma
        self.print_every = print_every

    def reinforce(self):
        scores_deque = deque(maxlen=100)
        scores = []
        for i_episode in range(1, self.episodes+1):
            saved_log_probs = []
            rewards = []
            state = self.env.reset()
            for t in range(max_t):
                action, log_prob = 
