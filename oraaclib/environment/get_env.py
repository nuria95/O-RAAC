import os

import d4rl
import gym
import h5py
from oraaclib.dataset.experience_replay import D4RL_Dataset
from oraaclib.environment.reward_wrappers import (RewardHighVelocity,
                                                  RewardUnhealthyPose)
from oraaclib.util.create_hdf5 import HDF5_Creator
from utils.utilities import get_keys

path_to_datasets = os.environ.get('D4RL_DATASET_DIR',
                                  os.path.expanduser('~/.d4rl/datasets'))


def get_gym_name(dataset_name):
    # Get v3 version of environments for extra information to be available
    if 'cheetah' in dataset_name:
        return 'HalfCheetah-v3'
    elif 'hopper' in dataset_name:
        return 'Hopper-v3'
    elif 'walker' in dataset_name:
        return 'Walker2d-v3'
    else:
        raise ValueError("{dataset_name} is not in D4RL")


def get_env(p, SEED=None, eval=False, reset_noise_scale=None,
            eval_terminate_when_unhealthy=True):

    terminate_when_unhealthy = False \
        if not eval_terminate_when_unhealthy else True
    # if not p.env.terminate_when_unhealthy \
    # or not eval_terminate_when_unhealthy else True

    env_d4rl = gym.make(p.env.name)
    dataset_name = env_d4rl.dataset_filepath[:-5]
    # Use v3 version of environments for extra information to be available
    kwargs = {'terminate_when_unhealthy': terminate_when_unhealthy} if \
        'cheetah' not in dataset_name else {}
    if reset_noise_scale is not None:
        kwargs['reset_noise_scale'] = reset_noise_scale
    env = gym.make(get_gym_name(dataset_name),
                   **kwargs
                   ).unwrapped
    if SEED is None:
        env.unwrapped.seed(seed=p.agent.SEED)
    else:
        env.unwrapped.seed(seed=SEED)

    if p.env.prob_vel_penal is not None and p.env.prob_vel_penal > 0:
        dict_env = {'prob_vel_penal': p.env.prob_vel_penal,
                    'cost_vel': p.env.cost_vel,
                    'max_vel': p.env.max_vel}

        fname = f'{dataset_name}_'\
            f'prob{dict_env["prob_vel_penal"]}_'\
            f'penal{dict_env["cost_vel"]}_'\
            f'maxvel{dict_env["max_vel"]}.hdf5'

        env = RewardHighVelocity(env, **dict_env)

    elif p.env.prob_pose_penal is not None and \
            p.env.prob_pose_penal > 0:
        dict_env = {'prob_pose_penal': p.env.prob_pose_penal,
                    'cost_pose': p.env.cost_pose,
                    }

        fname = f'{dataset_name}_'\
                f'prob{dict_env["prob_pose_penal"]}_'\
                f'penal{dict_env["cost_pose"]}_'\
                'pose.hdf5'
        env = RewardUnhealthyPose(env, **dict_env)

    else:
        fname = env_d4rl.dataset_filepath

    # Get dataset for training:
    if not eval:
        h5py_path = os.path.join(path_to_datasets, fname)
        if not os.path.exists(h5py_path):
            print(f'\n{h5py_path} doesn\'t exist.')
            creator = HDF5_Creator(d4rl_env_name=p.env.name,
                                   properties_env=dict_env,
                                   fname=fname)
            creator.create_hdf5_file()
        dataset_file = h5py.File(h5py_path, 'r')
        data = {k: dataset_file[k][:] for k in get_keys(dataset_file)}
        dataset_file.close()

        print(f'\nChecking dataset {h5py_path} is correct...')
        # Use get_dataset method from OfflineEnv Class in
        # D4RL to check dataset
        env_d4rl.get_dataset(h5path=h5py_path)
        print('Dataset correct\n')

        dataset = D4RL_Dataset(data)
    if eval:
        dataset = None

    return env, dataset
