import numpy as np
import gym
from gym import wrappers
from keras.models import Model
from keras.layers import Dense, Flatten, Input, concatenate
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

def build_actor_model(num_action, observation_shape):
    action_input = Input(shape=(1,)+observation_shape)
    x = Flatten()(action_input)
    x = Dense(16, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(num_action, activation="linear")(x)
    actor = Model(inputs=action_input, outputs=x)
    return actor

def build_critic_model(num_action, observation_shape):
    action_input = Input(shape=(num_action,))
    observation_input = Input(shape=(1,)+observation_shape)
    flattened_observation = Flatten()(observation_input)
    x = concatenate([action_input, flattened_observation])
    x = Dense(32, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(1, activation="linear")(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    return (critic, action_input)

def build_agent(num_action, observation_shape):
    actor = build_actor_model(num_action, observation_shape)
    critic, critic_action_input = build_critic_model(num_action, observation_shape)
    memory = SequentialMemory(limit=10**6, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=num_action, theta=0.15, mu=0, sigma=0.3)
    agent = DDPGAgent(
        num_action,
        actor,
        critic,
        critic_action_input,
        memory,
        random_process=random_process
    )
    return agent

def run():
    env = gym.make("MountainCarContinuous-v0")
    env = wrappers.Monitor(env, directory="/tmp/mountain-car-continuous-v0", force=True)
    print("Action Space: %s" % env.action_space)
    print("Observation Space: %s" % env.observation_space)
    agent = build_agent(env.action_space.shape[0], env.observation_space.shape)
    agent.compile(optimizer="nadam", metrics=["mae"])
    agent.fit(env, nb_steps=10**6, visualize=True, verbose=1, nb_max_episode_steps=200000)
    agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)

if __name__ == "__main__":
    run()
    
