import qLearning
from env import TicTacToe

env = TicTacToe()
agent1 = qLearning.QLearningAgent(env.action_num, 0.2, 0.9, 0.95, 0.01, 300, True, "ag1")
agent2 = qLearning.QLearningAgent(env.action_num, 0.2, 0.9, 0.95, 0.01, 300, True, "ag2")

agent1.mark = 1
agent2.mark = 2

period_train_ag1 = 270
period_train_ag2 = 270

period_count = 500
test_num = 20

use_last_module:bool = True