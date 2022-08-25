import qLearning
from env import TicTacToe
import os
import time

env = TicTacToe()
agent1 = qLearning.QLearningAgent(env.action_num, 0, 0, 0, 0, 0, False, "ag1")
agent2 = qLearning.QLearningAgent(env.action_num, 0, 0, 0, 0, 0, False, "ag2")
curr_path = os.path.dirname(os.path.abspath(__file__))
if __name__ == "__main__":
    agent1.load_module(curr_path + "/QTables/AG1module.pkl")
    agent2.load_module(curr_path + "/QTables/AG2module.pkl")
    state = [0] * 9
    while True:
        action = agent1.choose_action(state)
        next_state, reward, done = env.step(action, 1)
        state = next_state
        #print map
        for i in range(3):
            for j in range(3):
                print(str(env.map[i * 3 + j]) + ' ', end='')
            print()
        print()
        if done:
            if reward == -999:
                print("Agent 1 Cheated")
            elif reward == 70:
                print("Agent 1 win")
            elif reward == 50:
                print("Tie")
            break
        time.sleep(0.1)
        action = agent2.choose_action(state)
        next_state, reward, done = env.step(action, 2)
        state = next_state
        #print map
        for i in range(3):
            for j in range(3):
                print(str(env.map[i * 3 + j]) + ' ', end='')
            print()
        print()
        if done:
            if reward == -999:
                print("Agent 2 Cheated")
            elif reward == 70:
                print("Agent 2 win")
            elif reward == 50:
                print("Tie")
            break
        time.sleep(0.1)