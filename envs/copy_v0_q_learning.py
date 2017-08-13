import numpy as np
import gym
from gym import wrappers

def run(alpha=0.3, gamma=0.9):
    Q = {}
    env = gym.make("Copy-v0")
    env = wrappers.Monitor(env, '/tmp/copy-v0-q-learning', force=True)
    Gs = []
    for episode in range(10**6):
        x = env.reset()
        X, A, R = [], [], [] # States, Actions, Rewards
        done = False
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
        x, a, r = X[-1], A[-1], R[-1]
        Q[x][a] += alpha * (r - Q[x][a])
        for t in range(T-2, -1, -1):
            x, nx, a, r = X[t], X[t+1], A[t], R[t]
            Q[x][a] += alpha * (r + gamma * np.max(Q[nx].values()) - Q[x][a])
        G = sum(R) # Revenue 
        print "Episode: %d, Revenue: %d" % (episode, G)
        Gs.append(G)
        if np.mean(Gs[-100:]) > 25.0:
            break

if __name__ == "__main__":
    run()

