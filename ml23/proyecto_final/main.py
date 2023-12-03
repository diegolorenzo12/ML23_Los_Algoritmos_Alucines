import gymnasium as gym
import numpy as np
from network import Agent

env = gym.make("MountainCar-v0", render_mode="human")

gamma = 0.99
epsilon = 1.0
lr = 0.0001
input_dims = [2]
n_actions = 3
batch_size = 64
max_actions = 1000

agent = Agent(gamma=gamma, epsilon=epsilon, lr=lr, input_dims=input_dims,
              batch_size=batch_size, n_actions=n_actions)

num_episodes = 500
for episode in range(num_episodes):
    done = False
    observation = env.reset()[0]
    total_reward = 0
    actions_taken = 0

    while not done:
        action = agent.choose_action(observation)
        new_observation, reward, done, _, _ = env.step(action) #_ representa info de debug
        total_reward += reward
        agent.store_transition(observation, action, reward, new_observation, done)
        agent.learn()
        observation = new_observation
        actions_taken += 1

        if(actions_taken > max_actions):
            done = True

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

test_episodes = 10
for _ in range(test_episodes):
    observation = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.choose_action(observation)
        observation, reward, done, _ = env.step(action)
        total_reward += reward

    print(f"Test Episode: {_}, Total Reward: {total_reward}")

env.close()