import numpy as np
import math
import torch
from collections import defaultdict

class QLearningAgent(object):
    def __init__(self, action_num, learn_rate, gamma,
                epsilon_start, epsilon_end, epsilon_decay, is_learn, name):
        #get arguments
        self.action_num = action_num
        self.learn_rate = learn_rate
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.is_learn = is_learn
        self.gamma = gamma

        self.epsilon = 0
        self.Q_table = defaultdict(lambda: np.zeros(action_num))
        self.sample_count = 0
        self.name = name
    
    def choose_action(self, state):
        #epsilon decay
        if self.is_learn:
            self.sample_count += 1
            self.epsilon = self.epsilon_end + \
                (self.epsilon_start - self.epsilon_end) * \
                    math.exp(-1. * self.sample_count / self.epsilon_decay)
            #print(self.name + " is Learning...")
        
        # epsilon-greedy
        max = np.max(self.Q_table[str(state)])
        #print('\n' + str(max != 0))
        if ((np.random.uniform(0, 1) > self.epsilon) or (not self.is_learn)) and max != 0:
            #print(self.name + str(self.is_learn))
            #print(self.name + " actions: ", self.Q_table[str(state)])
            #print(self.name + " action: ", np.argmax(self.Q_table[str(state)]))
            return np.argmax(self.Q_table[str(state)])
        else:
            #print(self.name + " greedy")
            return np.random.choice(self.action_num)

    def update(self, state, action, reward, next_state, done):
        if self.is_learn:
            Q_predict = self.Q_table[str(state)][action]
            if done:
                Q_target = reward
            else:
                #print('\n' + self.name + ' ' + str(self.Q_table[next_state]) + str(np.max(self.Q_table[next_state])))
                Q_target = reward + self.gamma * np.max(self.Q_table[next_state])
                #print("State: " + str(state))
                #print("Q_target: " + str(Q_target))
                #print("target: " + str(Q_target))
            self.Q_table[str(state)][action] += self.learn_rate * (Q_target - Q_predict)
            #print("change: " + str(self.Q_table[str(state)][action]))

    def save_module(self, path):
        import dill
        torch.save(obj=self.Q_table, f=path, pickle_module=dill)
        print("Module saved")
    
    def load_module(self, path):
        import dill
        self.Q_table = torch.load(f=path, pickle_module=dill)
        print("Module loaded")