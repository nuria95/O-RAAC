import os

import gym
import numpy as np
import d4rl  # Import required to register environments
from rllib.dataset.utilities import DatasetWriter
from rllib.environment.reward_wrappers import (RewardHighVelocity,
                                               RewardUnhealthyPose)
import torch

path_to_datasets = os.environ.get('D4RL_DATASET_DIR',
                                  os.path.expanduser('~/.d4rl/datasets'))


def get_gym_name(dataset_name):
    if 'cheetah' in dataset_name:
        return 'HalfCheetah-v3'
    elif 'hopper' in dataset_name:
        return 'Hopper-v3'
    elif 'walker' in dataset_name:
        return 'Walker2d-v3'
    else:
        raise ValueError("{dataset_name} is not in D4RL")


class HDF5_Creator():

    """Create a hdf5 file containing training data for offline RL algorithm
    with a new reward function.
    Original dataset is obtained from D4RL environemnt but new data
    is obtained by running version 3 of the same environment type to get
    additional information only provided in version 3.

    Parameters
    ----------
    d4rl_env_name: str
                Name of the original hdf5 provided by D4RL.
                Example: 'walker2d-expert-v0'
    properties_env: dict
            dictionary containing information about the new reward function

            For speed penalization:
            dict_env = {'prob_vel_penal': float,
                            'cost_vel': int,
                            'max_vel': float}
            For pose penalization:
            dict_env = {'prob_pose_penal':float,
                            'cost_pose': int,
                            }
    """

    def __init__(self, d4rl_env_name, properties_env=None, fname=None):
        self.env_d4rl = gym.make(d4rl_env_name)
        self.dataset = self.env_d4rl.get_dataset()
        self.writer = DatasetWriter(mujoco=True)

        env = gym.make(get_gym_name(d4rl_env_name)).unwrapped
        env.seed(10)
        torch.manual_seed(10)
        np.random.seed(10)
        self.actions = self.dataset['actions']
        self.obs = self.dataset['observations']
        self.rewards = self.dataset['rewards']
        self.dones = self.dataset['terminals']
        self.properties_env = properties_env

        if properties_env.get('cost_vel', False):
            self.env = RewardHighVelocity(env, **properties_env)
        elif properties_env.get('cost_pose', False):
            self.env = RewardUnhealthyPose(env, **properties_env)
        else:
            raise ValueError('No reward wrapper found')

        dataset_name = self.env_d4rl.dataset_filepath[:-5]

        self.h5py_name = f'{dataset_name}_'\
            f'prob{properties_env["prob_vel_penal"]}_'\
            f'penal{properties_env["cost_vel"]}_'\
            f'maxvel{properties_env["max_vel"]}.hdf5'

        assert fname == self.h5py_name, \
            f'Not same name for h5py file {fname} vs {self.h5py_name}'

    def get_state(self, i):
        pos_full = np.concatenate([[self.env.sim.data.qpos[0].copy()],
                                   self.obs[i][0:8]]
                                  )  # pos_full = [xpos, 8jointpos]
        vel_full = self.obs[i][8::]  # [9 jointvels]
        return pos_full, vel_full

    def create_hdf5_file(self):
        self.writer._reset_data()
        init_pos = np.concatenate([[0], self.obs[0][0:8]])
        init_vel = self.obs[0][8::]

        self.env.reset()
        print(len(init_pos), (len(init_vel)))  # check
        self.env.set_state(qpos=init_pos, qvel=init_vel)

        print('\n\n **** Creating new dataset...******\n\n')
        for i in range(len(self.actions)):
            pos_full, vel = self.get_state(i)  # contains xposition
            observation = np.concatenate(
                (pos_full[1:], vel)).ravel()  # remove xposition
            action = self.actions[i]
            if not i % 10000:
                print(f'Num datapoint {i}/{len(self.actions)}')

            # reset state to data, needed to run open-loop
            self.env.set_state(qpos=pos_full, qvel=vel)
            _, reward, _, info = self.env.step(self.actions[i])

            if self.properties_env is None:  # no probpenal
                r = self.rewards[i]
            else:
                r = reward
            self.writer.append_data(observation, action, r, self.dones[i],
                                    mujoco_env_data=[
                                    pos_full, vel], info=info)

        self.writer.write_dataset(fname=self.h5py_name)
        print(f'Checking dataset {self.h5py_name} is correct...')
        # Use get_dataset method from OfflineEnv Class in
        # D4RL to check dataset
        new_dataset = self.env_d4rl.get_dataset(h5path=self.h5py_name)

        for _ in range(10):
            random_datapoint = np.random.randint(0, len(self.dataset))
            assert all(self.dataset['actions'][random_datapoint] ==
                       new_dataset['actions'][random_datapoint]),\
                f'{self.h5py_name} is not correct. Not same actions!'
            assert all(self.dataset['observations'][random_datapoint] ==
                       new_dataset['observations'][random_datapoint]),\
                f'{self.h5py_name} is not correct. Not same observations!'

        print(f'Dataset correct. Dataset saved in {self.h5py_name}')


if __name__ == "__main__":
    name_env = ''  # 'halfcheetah-expert-v0'
    dict_env = {}  # {'prob_vel_penal': 0.05,
    # 'cost_vel': -60,
    # 'max_vel': 10}
    fname = ''
    # f'{path_to_datasets}/halfcheetah_expert_prob0.05_penal-60_maxvel10.hdf5'

    creator = HDF5_Creator(name_env, dict_env, fname)
    creator.create_hdf5_file()
