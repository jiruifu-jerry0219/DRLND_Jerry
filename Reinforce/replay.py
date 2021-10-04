import gym
gym.logger.set_level(40) # suppress warnings (please remove if gives error)
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


import torch
torch.manual_seed(0) # set random seed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import Policy
from Policy import Policy

env = gym.make('CartPole-v0')
env.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
agent = Policy().to(device)

# load the weights from file
agent.load_state_dict(torch.load('checkpoint.pth'))

state = env.reset()
n = 0
i = 0
while True:
    i += 1
    
    with torch.no_grad():
        action, _ = agent.act(state)
    env.render()
    n += 1
    next_state, reward, done, _ = env.step(action)
    state = next_state
    if i >= 1000:
        break
    if done:
        print('Finished the task in: {} steps'.format(n))
        n = 0
        state = env.reset()

env.close()
