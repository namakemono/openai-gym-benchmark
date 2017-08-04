import numpy as np
import gym
from gym import wrappers
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def build_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(200, kernel_initializer="he_normal", activation="relu", input_dim=input_dim))
    model.add(Dense(200, kernel_initializer="he_normal", activation="relu"))
    model.add(Dense(output_dim, kernel_initializer="he_normal", activation="sigmoid"))
    return model

def run():
    max_score = 200.0
    num_episodes = 3000
    env = gym.make("CartPole-v0")
    env = wrappers.Monitor(env, directory="/tmp/cartpole-v0", force=True)
    logger.info("Action Space: %s" % str(env.action_space))
    logger.info("Observation Space: %s" % str(env.observation_space))
    model = build_model(input_dim=env.observation_space.shape[0], output_dim=env.action_space.n)
    model.compile(loss="mse", optimizer="adam", metrics=["mae", "mse"])
    Gs = []
    for episode in range(num_episodes):
        x = env.reset()
        X, Q, A, R = [], [], [], []
        done = False
        while not done:
            q = model.predict(np.asarray([x]))[0]
            a = np.argmax(q)
            X.append(x)
            A.append(a)
            Q.append(q)
            x, r, done, info = env.step(a)
            R.append(r)
            # print episode, len(X), a, q, x, r, done, info
        alpha = 0.1
        T = len(X)
        G = np.sum(R)
        for t in range(T):
            a = A[t]
            Q[t][a] = (1-alpha) * Q[t][a] + alpha * (G / max_score)
        model.fit(np.asarray(X), np.asarray(Q), verbose=0, batch_size=T)
        logger.debug("Episode: %d/%d, Reward: %.2f" % (episode, num_episodes, G)) 
        Gs.append(G)
    logger.info("Average Reward: %.3f" % np.mean(Gs))
    env.close()

if __name__ == "__main__":
    run()
