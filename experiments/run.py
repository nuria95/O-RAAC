import json
from datetime import datetime as dt

import d4rl
import gym
import numpy as np
import torch
from oraaclib.agent import BCQ, BEAR
from oraaclib.environment import get_env
from oraaclib.util.logger import Logger
from oraaclib.util.rollout import oraac_rollout
from oraaclib.util.utilities import get_dict_hyperparams
from oraaclib.util.torch_utilities import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from utils.utilities import (dotdict,  get_names,
                             parse_args)

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
    with open('json_params/BEAR_BCQ/'+config_name) as f:
        params = json.load(f)
        for k, v in args.__dict__.items():
            if v is not None:
                for main_key in params.keys():
                    for dk in params[main_key].keys():
                        if k == dk:
                            params[main_key][dk] = v
    if args.env_name is not None:
        params['env']['name'] = args.env_name
    if args.agent_name is not None:
        params['agent']['name'] = args.agent_name

    p = dotdict(params)
    p.agent = dotdict(params["agent"])
    p.env = dotdict(params["env"])

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

early_stopper_rew = EarlyStopping(
    name_save=name_save, patience=20, verbose=True, delta=1,
    evol_type='Mean_cumreward', env_properties=p.env,
    agent_properties=p.agent)
early_stopper_var = EarlyStopping(
    name_save=name_save, patience=20, verbose=True, delta=1,
    evol_type='Cvar_cumreward', env_properties=p.env,
    agent_properties=p.agent)

if p.agent.name == 'BEAR':
    agent = BEAR(env=env, dataset=dataset, hyper_params=hyper_params,
                 eval=False,
                 early_stopper_rew=early_stopper_rew,
                 early_stopper_var=early_stopper_var,
                 logger=logger,
                 name_save=name_save)
elif p.agent.name == 'BCQ':
    agent = BCQ(env=env, dataset=dataset, hyper_params=hyper_params,
                eval=False,
                early_stopper_rew=early_stopper_rew,
                early_stopper_var=early_stopper_var,
                logger=logger,
                name_save=name_save)
else:
    raise ValueError(f'Agent "{p.agent.name}"" is not implemented. Only'
                     'BEAR and BCQ available')


gradient_steps = 10000000
max_episodes = 2000
MAX_EVAL_STEPS = 200

print(f'Start training algorithm with {agent.__class__.__name__} algorithm')

oraac_rollout(env, agent,
              gradient_steps=p.agent.GRADIENT_STEPS,
              max_episodes=p.agent.max_episodes,
              max_episode_steps=p.agent.MAX_EVAL_STEPS,
              eval_freq=p.agent.eval_freq,
              times_eval=20)
