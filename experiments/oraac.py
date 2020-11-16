import json
import os
# import warnings
from copy import deepcopy
from datetime import datetime as dt

import gym
import numpy as np
import torch
from rllib.agent import ORAAC
from rllib.environment import get_env
from rllib.util import get_dict_hyperparams, oraac_rollout
from rllib.util.logger import Logger
from rllib.util.neural_networks import (VAE, DeterministicNN_IQN, ORAAC_Actor,
                                        RAAC_Actor)
from rllib.util.torch_utilities import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from utils.utilities import (dotdict, find_file, get_names, get_names_eval,
                             parse_args, update_old_json_varnames)

# warnings.filterwarnings("ignore")

record_tensorboard = True  # if 'cluster' in os.getcwd() else False
save_model = True  # if 'cluster' in os.getcwd() else False
render_eval = False

args = parse_args()
if args.model_path:
    args.eval = True
if args.render:
    render_eval = True

torch.set_num_threads(args.num_threads)
config_name = args.config_name
date = dt.now().strftime("%Y_%m_%d_%H%M%S_%f")

if not args.eval:
    with open('json_params/ORAAC/'+config_name) as f:
        params = json.load(f)
        for k, v in args.__dict__.items():
            if v is not None:
                for main_key in params.keys():
                    for dk in params[main_key].keys():
                        if k == dk:
                            params[main_key][dk] = v
    if args.env_name is not None:
        params['env']['name'] = args.env_name

    p = dotdict(params)
    p.agent = dotdict(params["agent"])
    p.env = dotdict(params["env"])
    if args.ablated:
        p.agent.name = 'RAAC'

    # Defining name_file:
    name_file, name_tb, name_save, name_logger_folder = \
        get_names(p, args,
                  date,
                  record_tensorboard,
                  save_model)
    tb = SummaryWriter('{}'.format(name_tb)) if name_tb is not None else None
    logger = Logger(folder=name_logger_folder, name=f'{name_file}')
    torch.manual_seed(p.agent.SEED)
    np.random.seed(p.agent.SEED)
    hyper_params = get_dict_hyperparams(p)
    env, dataset = get_env(p)


if args.eval:
    folder_model = 'data_ICLR/*/'\
        '*/train/models/lamda*/'\
        'cvar*'
    model_dict_name = find_file(
        args.model_path, path_name=folder_model, extension='.tar')
    input(f"Loading: \n{model_dict_name}? \nEnter to continue")
    model_net = torch.load(model_dict_name)

    dict_env = {}
    for item in model_net['env_properties']:
        dict_env[item[0]] = item[1]
    p = dotdict({})
    p.env = dotdict(dict_env)

    dict_agent = {}
    for item in model_net['agent_properties']:
        dict_agent[item[0]] = item[1]
    p.agent = dotdict(dict_agent)
    update_old_json_varnames(p)

    seed = args.SEED if args.SEED is not None else 100
    env, _ = get_env(p, seed, eval=True, eval_terminate_when_unhealthy=True)

    name_file = model_dict_name[:-4].split('/')[-1]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.record:
        render_eval = False
        args.numexp = 1
        env = gym.wrappers.Monitor(
            env, directory=f'recording/ICLR_reviews/{p.agent.name}/'
            f'{p.env.name}/{name_file}', force=True, mode='evaluation')

try:
    dim_action = env.action_space.shape[0]
    num_actions = -1
except IndexError:
    dim_action = 1
    num_actions = env.action_space.n

try:
    dim_state = env.observation_space.shape[0]
    num_states = -1
except IndexError:
    dim_state = 1
    # Add an extra final state as terminal state.
    num_states = env.observation_space.n + 1

print('*****Environment properties*****\n')
print('Name', env.unwrapped.spec.id)
print('Action dimension', dim_action)
print('Num of possible actions', num_actions)
print('Action space', env.action_space.shape)
print('lims action space', env.action_space.low, env.action_space.high)
print('State dimension', dim_state)
print('Num of possible states', num_states)
print('Max episode steps in theory', env.spec.max_episode_steps)
print('Max reward threshold', env.spec.reward_threshold)

if p.agent.name == 'RAAC':
    policy = RAAC_Actor(dim_state=dim_state, dim_action=dim_action,
                        max_action=env.action_space.high[0],
                        phi=1-p.agent.lamda, tau=p.agent.TARGET_UPDATE_TAU)
else:
    policy = ORAAC_Actor(dim_state=dim_state, dim_action=dim_action,
                         max_action=env.action_space.high[0],
                         phi=1-p.agent.lamda, tau=p.agent.TARGET_UPDATE_TAU)


vae = VAE(dim_state=dim_state, dim_action=dim_action,
          latent_dim=dim_action*2, max_action=env.action_space.high[0])

if not args.eval:
    critic = DeterministicNN_IQN(dim_state=dim_state,
                                 dim_action=dim_action,
                                 layers_state=p.agent.HIDDEN_UNITS_STATE,
                                 layers_action=p.agent.HIDDEN_UNITS_ACTION,
                                 layers_f=p.agent.HIDDEN_UNITS_F,
                                 tau_embed_dim=p.agent.TAU_EMBEDDING_DIM,
                                 embedding_dim=p.agent.EMBEDDING_DIM,
                                 tau=p.agent.TARGET_UPDATE_TAU)

    target_policy = deepcopy(policy)
    target_critic = deepcopy(critic)

    early_stopper_rew = EarlyStopping(
        name_save=name_save, patience=20, verbose=True, delta=1,
        evol_type='Mean_cumreward', env_properties=p.env,
        agent_properties=p.agent)
    early_stopper_var = EarlyStopping(
        name_save=name_save, patience=20, verbose=True, delta=1,
        evol_type='Cvar_cumreward', env_properties=p.env,
        agent_properties=p.agent)

    agent = ORAAC(env, policy, critic, target_policy, target_critic,
                  hyper_params,
                  dataset,
                  tb=tb, vae=vae, logger=logger,
                  save_model=save_model,
                  early_stopper_rew=early_stopper_rew,
                  early_stopper_var=early_stopper_var, name_save=name_save)

    oraac_rollout(env, agent,
                  gradient_steps=p.agent.GRADIENT_STEPS,
                  max_episodes=p.agent.max_episodes,
                  max_episode_steps=p.agent.MAX_EVAL_STEPS,
                  eval_freq=p.agent.eval_freq,
                  times_eval=20)

if args.eval:
    policy.load_state_dict(model_net['actor'])
    vae.load_state_dict(model_net['vae'])
    critic = target_critic = target_policy = None
    name_file = model_dict_name[:-4].split('/')[-1]
    name_logger_folder = get_names_eval(p)
    if not os.path.exists(name_logger_folder):
        os.makedirs(name_logger_folder)
    name_logger = f'{name_file}_{date}'

    logger = Logger(
        folder=name_logger_folder, name=name_logger)

    agent = ORAAC(env, policy, critic, target_policy, target_critic,
                  dataset=None, eval=True, logger=logger, vae=vae)

    print('\nEvaluating model....')
    max_episode_steps = 200 if 'Cheetah' in env.name else 500

    agent.evaluate_model(max_episode_steps=max_episode_steps,
                         render=render_eval,
                         times_eval=args.numexp)

    print(f'Logged data saved in {logger.folder}/{logger.name}.json')
    env.close()
