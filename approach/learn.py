import numpy as np
import gymnasium as gym
import tensorflow as tf

from agent import Agent as ApproachAgent
from env import Env as ApproachEnv

env = ApproachEnv(True)
agent = ApproachAgent(env)

# Number of episodes
n_episodes = 3

for episode in range(n_episodes):
    state = env.reset()
    rewards = []
    log_probs = []
    done = False

    while not done:
        # action = agent.get_action(state)
        # next_state, reward, done, info = env.step(action)
        # log_prob = tf.math.log(agent.policy(tf.convert_to_tensor([state], dtype=tf.float32))[0, action])
        # log_probs.append(log_prob)
        # rewards.append(reward)
        # state = next_state

        next_state, reward, done, info = env.step(None)
        pass

    # agent.update_policy(rewards, log_probs)

    if episode % 1000 == 0:
        print(f"Episode: {episode}")

env.close()