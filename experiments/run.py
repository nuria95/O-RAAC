import gym
import d4rl
from rllib.agent import DDPG, BEAR, BCQ
from rllib.dataset.experience_replay import D4RL_Dataset
from rllib.util.rollout import oraac_rollout

hyper_params = dict()
env = gym.make("halfcheetah-medium-v0")
dataset = D4RL_Dataset(env.get_dataset())

agent = BEAR(env=env, dataset=dataset, hyper_params=hyper_params, eval=False)

gradient_steps = 10000000
max_episodes = 2000
MAX_EVAL_STEPS = 200

oraac_rollout(env, agent,
              gradient_steps=gradient_steps,
              max_episodes=max_episodes,
              max_episode_steps=MAX_EVAL_STEPS,
              eval_freq=20,
              times_eval=20)
