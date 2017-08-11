import numpy as np
import gym
from gym import wrappers

def run():
    env = gym.make('Copy-v0')
    env = wrappers.Monitor(env, '/tmp/copy-v0', force=True)
    Gs = []
    for episode in range(1000):
        x = env.reset()
        G = 0
        for t in range(100):
            a = (1,1, x)
            x, r, done, _ = env.step(a)
            G += r
            if done:
                Gs.append(G)
                break
        score = np.mean(Gs[-100:])
        print("Episode: %3d, Score: %.3f" % (episode, score))
        if score > 25:
            break


if __name__ == "__main__":
    run()

