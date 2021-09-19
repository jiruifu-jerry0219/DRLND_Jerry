import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        # initialize the Q table
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        
        self.env = None
        self.i_ep = 0
        # define learning rate, discount factor and epsilon
        self.eps = 0
        self.alpha = 0.5
        self.gamma = 0.99
        
    def q_function(self, state, action, value):
        self.Q[state][action] = valuie
        
    def episode(self, episode):
        self.i_ep = episode
    
    def env(self, env):
        self.env = env
    
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        self.eps = 1 / self.i_ep
        
        if np.random.random() > self.eps:
            return np.argmax(self.Q[state])
        else:
            return np.random.choice(np.arange(self.env.action_space.n))

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        current = self.Q[state][action]
        Qsa_next = np.max(self.Q[next_state])
        target = reward + (self.gamma * Qsa_next)
        new_value = current + (self.alpha * (target - current))
        self.Q[state][action] = new_value