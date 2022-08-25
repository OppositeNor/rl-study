import numpy as np
import os
    


class TicTacToe(object):
    def __init__(self):
        self.action_num = 9
        self.map = np.array(np.zeros(9))
        self.step_moved = 0
        self.win_reward = 100
        self.cheat_reward = -999
        self.tie_reward = 50
        self.step_reward = -1
        self.lost_reward = -100

    def reset(self):
        self.map = np.array(np.zeros(9))
        self.step_moved = 0
    
    #returns next_state, reward, done
    def step(self, action, mark):
        self.step_moved += 1
        if self.map[action] != 0:
            #cheated, punish
            self.map[action] = mark
            next_state = str(self.map)
            #print(f"{mark} cheated")
            reward = self.cheat_reward
            done = True
            return next_state, reward, done
        
        self.map[action] = mark
        next_state = str(self.map)
        if self.check_if_win(mark):
            #win reward
            reward = self.win_reward
            done = True
            #print(f"{mark} wins")
            return next_state, reward, done
        
        if self.step_moved == 9:
            #tie reward
            reward = self.tie_reward
            done = True
            #print("  tie", end='')
            return next_state, reward, done

        return next_state, self.step_reward, False
    

    def check_if_win(self, mark):
        
        if (np.array([mark, mark, mark]) == self.map[0:3]).all() or \
        (np.array([mark, mark, mark]) == self.map[3:6]).all() or\
        (np.array([mark, mark, mark]) == self.map[6:9]).all() or\
            (mark == self.map[0] and mark == self.map[4] and mark == self.map[8]) or \
            (mark == self.map[2] and mark == self.map[4] and mark == self.map[6]):
            return True
        for i in range(3):
            if mark == self.map[i] and mark == self.map[i + 3] and mark == self.map[i + 6]:
                #print("vert")
                return True
        return False

        
