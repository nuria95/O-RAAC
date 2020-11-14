import argparse
import fnmatch
import glob
import os

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', action="store", type=str,
                        default='Params_cheetah.json')
    parser.add_argument('--num_threads', action="store", type=int, default=1)
    parser.add_argument('--device', action="store", type=str)

    parser.add_argument('--SEED', action="store", type=int)
    parser.add_argument('--N_QUANTILES_POLICY', action="store", type=int)
    parser.add_argument('--N_QUANTILES_CURR', action="store", type=int)
    parser.add_argument('--TAU_EMBEDDING_DIM', action="store", type=int)
    parser.add_argument('--LEARNING_RATE_ACTOR', action="store", type=float)
    parser.add_argument('--LEARNING_RATE_CRITIC', action="store", type=float)
    parser.add_argument('--noise_eps_start', action="store", type=float)
    parser.add_argument('--lamda', action="store", type=float)
    parser.add_argument('--TARGET_UPDATE_TAU', action="store", type=float)

    parser.add_argument('--cvar', action="store", type=float)
    parser.add_argument('--prob_vel_penal', action="store", type=float)
    parser.add_argument('--max_vel', action="store", type=int)
    parser.add_argument('--cost_vel', action="store", type=int)
    parser.add_argument('--max_episodes', action="store", type=int)
    parser.add_argument('--eval_freq', action="store", type=int)

    parser.add_argument('--prob_pose_penal', action="store", type=float)
    parser.add_argument('--cost_pose', action="store", type=int)
    parser.add_argument('--env_name', action="store", type=str)
    parser.add_argument('--noise', default=False, action='store_true')
    parser.add_argument('--save_log', default=False, action='store_true')

    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--retrain', default=False, action='store_true')
    parser.add_argument('--model_path', action="store", type=str)
    parser.add_argument('--numexp', action="store", type=int, default=1)
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--record', default=False, action='store_true')
    parser.add_argument('--ablated', default=False, action='store_true')

    args = parser.parse_args()

    return args


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def find_file(name, path_name, extension):
    # Returns a list of names in list files.
    found_files = [None]
    for filename in glob.iglob(f'{path_name}/**/*',
                               recursive=True):

        if fnmatch.fnmatch(filename, '*'+name+'*'+extension):
            found_files.append(filename)
    assert_text = f'File {name}{extension} not found in {path_name}'
    assert any(found_files), assert_text
    if len(found_files) > 2:  # count the None
        for num, name in enumerate(found_files, start=0):
            print(f"{name}: {num}\n")
        num = int(input('Which file do you want?(num 1 is the one after None'))
        found_file = found_files[num]
    else:
        found_file = found_files[1]
    return found_file


def get_names(p, args, date, record_tensorboard, save_model):

    inf_penal = f'_velpenal{np.abs(p.env.cost_vel)}_' \
                f'prob{p.env.prob_vel_penal}' \
                if p.env.prob_vel_penal is not None else ''
    if inf_penal == '':
        inf_penal = f'_anglepenal{np.abs(p.env.cost_pose)}_' \
                    f'prob{p.env.prob_pose_penal}' \
                    if p.env.prob_pose_penal is not None else ''
    inf_cvar = f'_cvar{p.agent.cvar}'
    inf_seed = f'_seed{p.agent.SEED}'
    inf_lamda = f'_lamda{p.agent.lamda}'
    inf_tau_update = f'_tau{p.agent.TARGET_UPDATE_TAU}'

    name_file = '{}{}{}{}{}{}'.format(
        date, inf_penal, inf_cvar, inf_seed, inf_lamda, inf_tau_update)

    if record_tensorboard:
        name_tb = f'data_ICLR/{p.agent.name}/{p.env.name}/'\
            f'train/runs/lamda{p.agent.lamda}/{name_file}'
    else:
        name_tb = None

    if save_model:
        save_directory = f'data_ICLR/{p.agent.name}/'\
            f'{p.env.name}/train/models/lamda{p.agent.lamda}/'\
            f'cvar{p.agent.cvar}'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        name_save = os.path.join(save_directory, name_file)
    else:
        name_save = None

    name_logger_folder = (f'data_ICLR/{p.agent.name}/'
                          f'{p.env.name}/train/statistics/lamda{p.agent.lamda}/cvar{p.agent.cvar}')

    return name_file, name_tb, name_save, name_logger_folder


def get_names_eval(p):

    name_logger_folder = (f'data_ICLR/{p.agent.name}/eval/'
                          f'{p.env.name}/statistics/lamda{p.agent.lamda}/'
                          f'cvar{p.agent.cvar}')

    return name_logger_folder


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)
    h5file.visititems(visitor)
    return keys


def update_old_json_varnames(p):
    if p.env.prob_v_penal is not None:
        p.env.prob_vel_penal = p.env.prob_v_penal
        p.env.cost_vel = p.env.cost_high_vel
    if p.env.prob_unhealthy_penal is not None:
        p.env.prob_pose_penal = p.env.prob_unhealthy_penal
        p.env.cost_pose = p.env.cost_unhealthy
    if p.agent.name == 'O_DDACVAR':
        p.agent.name = 'O_RAAC'
