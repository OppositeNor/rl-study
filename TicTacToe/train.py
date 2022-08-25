import qLearning
import os
from env import TicTacToe
import matplotlib.pyplot as plt
from preferences import *


curr_path = os.path.dirname(os.path.abspath(__file__))
def save_cheat_data(mark, state, action, next_state):
    file = open(curr_path + "/CheatData/Cheat" + str(mark) + ".txt", 'a')
    file.write("\nState: " + str(state) + "\nAction: " + str(action) + "\nNext State: " + str(next_state) + "\n")
    file.close()

def write_Qt_in_file():
    file = open(curr_path + "/table1.txt", 'w')
    file.write(str(agent1.Q_table))
    file.close()
    file = open(curr_path + "/table2.txt", 'w')
    file.write(str(agent2.Q_table))
    file.close()

test_agent1 = qLearning.QLearningAgent(env.action_num, 0, 0, 0, 0, 0, False, "t-ag1")
test_agent2 = qLearning.QLearningAgent(env.action_num, 0, 0, 0, 0, 0, False, "t-ag2")

ag1_cheat_num_rec = []
ag2_cheat_num_rec = []
ag1_win_num_rec = []
ag2_win_num_rec = []
tie_num_rec = []

def test():
    print("Test start")
    state = [0] * 9
    test_agent1.load_module(curr_path + "/QTables/AG1module.pkl")
    test_agent2.load_module(curr_path + "/QTables/AG2module.pkl")
    ag1_cheat_num = 0
    ag2_cheat_num = 0
    ag1_win_num = 0
    ag2_win_num = 0
    tie_num = 0
    for i in range(test_num):
        env.reset()
        while True:
            action = test_agent1.choose_action(state)
            next_state, reward, done = env.step(action, agent1.mark)
            if done:
                if reward == -999:
                    ag1_cheat_num += 1
                    save_cheat_data(agent1.mark, state, action, next_state)
                    print("Agent 1 Cheated")
                elif reward == 100:
                    ag1_win_num += 1
                    print("Agent 1 win")
                elif reward == 50:
                    tie_num += 1
                    print("Tie")
                break
            state = next_state
            action = test_agent2.choose_action(state)
            next_state, reward, done = env.step(action, agent2.mark)
            if done:
                if reward == -999:
                    ag2_cheat_num += 1
                    save_cheat_data(agent2.mark, state, action, next_state)
                    print("Agent 2 Cheated")
                elif reward == 100:
                    ag2_win_num += 1
                    print("Agent 2 win")
                elif reward == 50:
                    tie_num += 1
                    print("Tie")
                break
            state = next_state
    ag1_cheat_num_rec.append(ag1_cheat_num)
    ag2_cheat_num_rec.append(ag2_cheat_num)
    ag1_win_num_rec.append(ag1_win_num)
    ag2_win_num_rec.append(ag2_win_num)
    tie_num_rec.append(tie_num)


if __name__ == "__main__":
    if use_last_module:
        agent1.load_module(curr_path + "/QTables/AG1module.pkl")
        agent2.load_module(curr_path + "/QTables/AG2module.pkl")
    write_Qt_in_file()
    for j in range(period_count):
        #train1
        agent1.is_learn = True
        agent2.is_learn = False
        count = 0
        agent1.sample_count = 0
        agent2.sample_count = 0
        for i in range(period_train_ag1):
            env.reset()
            state1 = [0] * 9
            state2 = [0] * 9
            is_first = True
            count = 0
            while True:
                count += 1
                #agent1 choose action
                ag1_action = agent1.choose_action(state1)
                #print(ag1_action)
                next_state, reward1, done = env.step(ag1_action, agent1.mark)
                #print(state1)
                state2 = next_state
                #agent 2 choose action
                ag2_action = agent2.choose_action(state2)
                next_state, reward2, done = env.step(ag2_action, agent2.mark)
                #update agent 1, from state1 to next_state
                #print(state1, next_state)
                
                if done:
                    if reward1 == env.cheat_reward:
                        #save_cheat_data(1, state1, ag1_action, next_state)
                        pass
                    elif reward2 == env.win_reward:
                        reward1 = env.lost_reward
                    agent1.update(state1, ag1_action, reward1, next_state, done)
                    break
                agent1.update(state1, ag1_action, reward1, next_state, done)
                is_first = False

                state1 = next_state
                #print(env.map)
            print(f"\r round {i} finished", end = '')
        agent1.is_learn = False
        agent2.is_learn = True
        print()
        agent1.sample_count = 0
        agent2.sample_count = 0
        #train 2
        for i in range(period_train_ag2):
            env.reset()
            state1 = [0] * 9
            state2 = [0] * 9
            is_first = True

            
            while True:
                #agent1 choose action
                ag1_action = agent1.choose_action(state1)
                #print(ag1_action)
                next_state, reward1, done = env.step(ag1_action, agent1.mark)
                #print(state1)
                #update agent 2, from state2 to next_state
                #if not is_first:
                    #agent2.update(state2, ag2_action, reward2, next_state, done)
                if done:
                    if reward2 == env.cheat_reward:
                        #save_cheat_data(2, state2, ag2_action, next_state)
                        pass
                    elif reward1 == env.win_reward:
                        reward2 = env.lost_reward
                    if reward1 == env.tie_reward:
                        reward2 = env.tie_reward
                    agent2.update(state2, ag2_action, reward2, next_state, done)
                    break
                if not is_first:
                    agent2.update(state2, ag2_action, reward2, next_state, done)
                state2 = next_state
                #agent 2 choose action
                ag2_action = agent2.choose_action(state2)
                next_state, reward2, done = env.step(ag2_action, agent2.mark)
                #print(state1, next_state)
                state1 = next_state
                is_first = False
            print(f"\r round {i} finished", end = '')
            
        print(f"\nperiod {j} finished")
        agent1.save_module(curr_path + "/QTables/AG1module.pkl")
        agent2.save_module(curr_path + "/QTables/AG2module.pkl")
        test()
            #print(f"round {i} finish {env.map}")
    write_Qt_in_file()
    plt.plot(ag1_cheat_num_rec, label="ag1_cheat_num_rec")
    plt.plot(ag2_cheat_num_rec, label="ag2_cheat_num_rec")
    plt.plot(ag1_win_num_rec, label="ag1_win_num_rec")
    plt.plot(ag2_win_num_rec, label="ag2_win_num_rec")
    plt.plot(tie_num_rec, label="tie_num_rec")
    plt.legend()
    plt.show()

    #print(agent1.Q_table)
    #print(agent2.Q_table)