import gym
import argparse

parser = argparse.ArgumentParser(description='Upload your result to OpenAI Gym.')
parser.add_argument('--training-dir', type=str, required=True, help="A directory containing the results of a training run.")
parser.add_argument('--api-key', type=str, required=True, help="Your OpenAI API Key. Check https://gym.openai.com/users/<user>.")
args = parser.parse_args()
gym.upload(args.training_dir, api_key=args.api_key)
