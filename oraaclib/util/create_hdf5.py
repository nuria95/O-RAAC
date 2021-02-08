import os

import d4rl  # Import required to register environments
import gym
import numpy as np
import torch
from oraaclib.dataset.utilities import DatasetWriter
from oraaclib.environment.reward_wrappers import (RewardHighVelocity,
                                                  RewardUnhealthyPose)
from torch.distributions import Bernoulli

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
        self.d4rl_env_name = d4rl_env_name
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
        dataset_name = self.env_d4rl.dataset_filepath[:-5]

        if properties_env.get('cost_vel', False):
            self.env = RewardHighVelocity(env, **properties_env)
            self.h5py_name = f'{dataset_name}_'\
                f'prob{properties_env["prob_vel_penal"]}_'\
                f'penal{properties_env["cost_vel"]}_'\
                f'maxvel{properties_env["max_vel"]}.hdf5'

        elif properties_env.get('cost_pose', False):
            self.env = RewardUnhealthyPose(env, **properties_env)
            self.h5py_name = f'{dataset_name}_'\
                f'prob{properties_env["prob_pose_penal"]}_'\
                f'penal{properties_env["cost_pose"]}_'\
                'pose.hdf5'

        else:
            raise ValueError('No reward wrapper found')

        dataset_name = self.env_d4rl.dataset_filepath[:-5]

        assert fname == self.h5py_name, \
            f'Not same name for h5py file {fname} vs {self.h5py_name}'

    def create_hdf5_file(self):
        if 'cheetah' in self.d4rl_env_name:
            # Need to rerun environment since vel is not provided in obs:
            self.create_hdf5_file_cheetah()
        else:
            # Apply reward function on the reward
            self.create_hdf5_file_hopper_walker()
        self.check_data()

    def create_hdf5_file_hopper_walker(self):

        print('\n\n **** Creating new dataset...******\n\n')
        min_angle, max_angle = self.env.robust_angle_range
        penal_distr = Bernoulli(self.properties_env['prob_pose_penal'])

        for i in range(len(self.actions)):
            observation = self.obs[i]
            # differs from env.sim.data.qpos[1:3] in RewardWrapper
            # since in observation xposition (qpos[0]) has already
            # been excluded
            _, angle = observation[0], observation[1]
            robust_angle = min_angle < angle < max_angle
            penal = (~robust_angle) *\
                penal_distr.sample().item() * self.properties_env['cost_pose']
            r = penal + self.rewards[i]
            if not i % 10000:
                print(f'Num datapoint {i}/{len(self.actions)}')

            self.writer.append_data(observation, self.actions[i], r,
                                    self.dones[i])

        self.writer.write_dataset(fname=self.h5py_name)

    def get_state(self, i):
        pos_full = np.concatenate([[self.env.sim.data.qpos[0].copy()],
                                   self.obs[i][0:8]]
                                  )  # pos_full = [xpos, 8jointpos]
        vel_full = self.obs[i][8::]  # [9 jointvels]
        return pos_full, vel_full

    def create_hdf5_file_cheetah(self):
        self.writer._reset_data()
        init_pos = np.concatenate([[0], self.obs[0][0:8]])  # 5 for hopper
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

    def check_data(self):
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
    # To create the HDF5 file directly:
    dict_env = {
        "name": "walker2d-expert-v0",
        "prob_pose_penal": 0.15,
        "cost_pose": -30,
    }
    name_env = dict_env['name']
    env_d4rl = gym.make(name_env)
    dataset_name = env_d4rl.dataset_filepath[:-5]
    if dict_env.get('prob_vel_penal'):
        fname = f'{dataset_name}_'\
                f'prob{dict_env["prob_vel_penal"]}_'\
            f'penal{dict_env["cost_vel"]}_'\
            f'maxvel{dict_env["max_vel"]}.hdf5'
    else:
        fname = f'{dataset_name}_'\
                f'prob{dict_env["prob_pose_penal"]}_'\
            f'penal{dict_env["cost_pose"]}_'\
            f'pose.hdf5'

    creator = HDF5_Creator(name_env, dict_env, fname)
    creator.create_hdf5_file()
