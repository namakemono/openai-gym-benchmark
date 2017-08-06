# -*- coding: utf-8 -*-
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import numpy as np
import gym
from gym import wrappers

def run():
    env = gym.make("FrozenLake-v0")
    env = wrappers.Monitor(env, directory="/tmp/frozenlake-v0", force=True)
    logger.info("Observation Space: %d, Action Space: %d" % (env.observation_space.n, env.action_space.n))
    max_episode = 10000
    Q = np.zeros((env.observation_space.n, env.action_space.n)) # Action Values
    Gs = [] # Revenues
    best = -1
    for episode in range(max_episode):
        x = env.reset()
        X = [] # States
        done = False
        alpha, gamma = 0.03, 0.9
        while not done:
            if np.random.random() < 0.001:
                a = np.random.randint(env.action_space.n)
            else:
                a = np.argmax(Q[x,:])
            X.append(x)
            x, r, done, info = env.step(a)
            if done:
                r = (2*r - 1) * 100.0
            logger.debug("State: %d, Reward: %d, Done: %s, Info: %s" % (x, r, done, info))
            x_pre = X[-1]
            Q[x_pre,a] += alpha * (r + gamma * np.max(Q[x,:]) - Q[x_pre,a])
        Gs.append(int(r > 0))
        avg = np.mean(Gs[-100:])
        best = max(best, avg)
        logger.info("Episode: %d, End of turn: %d, Revenue: %.2f, Average: %.2f, Best: %.2f" % (episode, len(X), r, avg, best))

if __name__ == "__main__":
    run()
