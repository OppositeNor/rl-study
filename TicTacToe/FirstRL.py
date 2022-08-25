import gym
env = gym.make('CliffWalking-v0')
env.seed(1)
n_status = env.observation_space.n
n_actions = env.action_space.n
print(f"状态数：{n_status}, 动作数：{n_actions}")
state = env.reset()
print(f"初始状态：{state}")
