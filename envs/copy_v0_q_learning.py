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
        G = 0
        done = False
        while not done:
            if (np.random.random() < 0.01) or (not x in Q):
                a = env.action_space.sample()
            else:
                a = sorted(Q[x].items(), key=lambda _: -_[1])[0][0]
            nx, r, done, _ = env.step(a)
            if not x in Q:
                Q[x] = {a: 0}
            if not nx in Q:
                Q[nx] = {a: 0}
            if not a in Q[x]:
                Q[x][a] = 0
            G += r
            Q[x][a] += alpha * (r + gamma * np.max(Q[nx].values()) - Q[x][a])
            x = nx
        print "Episode: %d, Reward: %d" % (episode, G)
        Gs.append(G)
        if np.mean(Gs[-100:]) > 25.0:
            break

if __name__ == "__main__":
    run()

