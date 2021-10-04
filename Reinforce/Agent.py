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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
policy = Policy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

def main(env, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100):

    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes+1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma**i for i in range(len(rewards)+1)]
        R = sum([a*b for a,b in zip(discounts, rewards)])

        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()


        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f} \tCheckpoint Saved!'.format(i_episode, np.mean(scores_deque)))
            torch.save(policy.state_dict(), 'checkpoint.pth')
        if np.mean(scores_deque)>=195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f} \tCheckpoint Saved!'.format(i_episode-100, np.mean(scores_deque)))
            torch.save(policy.state_dict(), 'checkpoint.pth')
            break

    return scores

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    env.seed(0)
    print('observation space:', env.observation_space)
    print('action space:', env.action_space)
    scores = main(env)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
