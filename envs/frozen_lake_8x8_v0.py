# -*- coding: utf-8 -*-
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import numpy as np
import gym
from gym import wrappers
from keras.models import Sequential
from keras.layers import Dense

def build_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(input_dim, kernel_initializer="he_normal", activation="relu", input_shape=(input_dim, )))
    model.add(Dense(output_dim, kernel_initializer="he_normal", activation="softmax"))
    return model

def run():
    env = gym.make("FrozenLake8x8-v0")
    env = wrappers.Monitor(env, directory="/tmp/frozenlake8x8-v0", force=True)
    logger.info("Observation Space: %d, Action Space: %d" % (env.observation_space.n, env.action_space.n))
    max_episode = 100000
    Gs = [] # Revenues
    best = -1
    model = build_model(env.observation_space.n, env.action_space.n)
    model.compile(loss="categorical_crossentropy", optimizer="nadam", metrics=["mae", "mse"])
    for episode in range(max_episode):
        x = env.reset()
        X = [] # States
        A = [] # Actions
        Q = [] # Action Values
        done = False
        alpha, gamma = 0.03, 0.9
        while not done:
            v = np.zeros(env.observation_space.n)
            v[x] = 1
            q = model.predict(np.asarray([v]))[0]
            a = np.argmax(q)
            X.append(v)
            A.append(a)
            Q.append(q)
            x, r, done, info = env.step(a)
            if done:
                r = +1 if r > 0 else -1
                break
        T = len(X)
        for t in range(T-1):
            a = A[t]
            Q[t][a] = (1-alpha) * Q[t][a] + alpha * r
        model.fit(np.asarray(X), np.asarray(Q), verbose=0, batch_size=T)
        Gs.append(int(r > 0))
        avg = np.mean(Gs[-100:])
        best = max(best, avg)
        if best > 0.95:
            break
        logger.info("Episode: %d, End of turn: %d, Revenue: %.2f, Average: %.2f, Best: %.2f" % (episode, len(X), r, avg, best))

if __name__ == "__main__":
    run()
