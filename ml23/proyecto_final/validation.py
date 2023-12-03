import gymnasium as gym
import numpy as np
from network import Agent
import utils as util

env = gym.make("MountainCar-v0", render_mode="human")

gamma = 0.99
epsilon = 0.0
lr = 0.0001
input_dims = [2]
n_actions = 3
batch_size = 64
max_actions = 200
rewards = []

agent = Agent(gamma=gamma, epsilon=epsilon, lr=lr, input_dims=input_dims,
              batch_size=batch_size, n_actions=n_actions)

agent.load_model("model.bin")

num_episodes = 10
for episode in range(num_episodes):
    observation = env.reset()[0]
    done = False
    total_reward = 0
    actions = 0

    while not done:
        action = agent.choose_action(observation)
        new_observation, reward, done, _, _ = env.step(action)
        total_reward += reward
        observation = new_observation
        actions += 1

        if(actions >= max_actions):
            done = True

    rewards.append(total_reward)
    print(f"Test Episode: {episode}, Total Reward: {total_reward}")

util.plot_rewards(rewards, lr, max_actions)

env.close()