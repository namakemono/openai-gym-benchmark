import numpy as np
import gym
from gym import wrappers

def run(alpha=0.3, gamma=0.9):
    Q = {}
    env = gym.make("Copy-v0")
    env = wrappers.Monitor(env, '/tmp/copy-v0-sarsa', force=True)
    Gs = []
    for episode in range(10**6):
        x = env.reset()
        done = False
        X, A, R = [], [], [] # States, Actions, Rewards
        while not done:
            if (np.random.random() < 0.01) or (not x in Q):
                a = env.action_space.sample()
            else:
                a = sorted(Q[x].items(), key=lambda _: -_[1])[0][0]
            X.append(x)
            A.append(a)
            if not x in Q:
                Q[x] = {}
            if not a in Q[x]:
                Q[x][a] = 0
            x, r, done, _ = env.step(a)
            R.append(r)
        T = len(X)
        for t in range(T-1, -1, -1):
            if t == T-1:
                x, a, r = X[t], A[t], R[t]
                Q[x][a] += alpha * (r - Q[x][a])
            else:
                x, nx, a, na, r = X[t], X[t+1], A[t], A[t+1], R[t]
                Q[x][a] += alpha * (r + gamma * Q[nx][na] - Q[x][a])
        G = sum(R) # Revenue
        print "Episode: %d, Reward: %d" % (episode, G)
        Gs.append(G)
        if np.mean(Gs[-100:]) > 25.0:
            break

if __name__ == "__main__":
    run()

