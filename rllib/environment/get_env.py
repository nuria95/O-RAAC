import os

import d4rl
import gym
import h5py
from rllib.dataset.experience_replay import D4RL_Dataset
from utils.utilities import get_keys

from .reward_wrappers import RewardHighVelocity, RewardUnhealthyPose


def get_env(p, SEED=None, eval=False, noise=False, sigma_noise=None,
            eval_terminate_when_unhealthy=True):
    if 'O_RAAC' in p.agent.name:
        path = os.environ.get('D4RL_DATASET_DIR',
                              os.path.expanduser('~/.d4rl/datasets'))

        terminate_when_unhealthy = False \
            if not p.env.terminate_when_unhealthy \
            or not eval_terminate_when_unhealthy else True

        # Expertise level of Behavior policy:
        expertise_level = None
        expertise_levels = ['expert', 'medium']
        for level in expertise_levels:
            if level in p.env.name:
                expertise_level = level
        assert expertise_level is not None, 'No expertise level found in \
                                            datasets'

        if 'cheetah' in p.env.name:
            # use normalEnv to be able to use extra info
            env = gym.make('HalfCheetah-v3').unwrapped
            
            if SEED is None:
                env.unwrapped.seed(seed=p.agent.SEED)
            else:
                env.unwrapped.seed(seed=SEED)

            if p.env.prob_vel_penal is not None and p.env.prob_vel_penal > 0:
                dict_env = {'prob_vel_penal': p.env.prob_vel_penal,
                            'cost_vel': p.env.cost_vel,
                            'max_vel': p.env.max_vel}
                # Get dataset:
                fname = f'halfcheetah_{expertise_level}_'\
                    f'redone_prob{dict_env["prob_vel_penal"]}'\
                    f'_penal{dict_env["cost_vel"]}_'\
                    f'maxvel{dict_env["max_vel"]}.hdf5'

                env = RewardHighVelocity(env, **dict_env)
            else:
                fname = f'halfcheetah_{expertise_level}_redone.hdf5'

            if not eval:
                h5path = os.path.join(
                    path, fname)
                dataset_file = h5py.File(h5path, 'r')
                data = {k: dataset_file[k][:] for k in get_keys(dataset_file)}
                dataset_file.close()
                print(f'Checking dataset {h5path} is correct')
                # Use get_dataset method from OfflineEnv Class in
                # D4RL to check dataset
                env_test = gym.make(p.env.name)
                env_test.get_dataset(h5path=h5path)

        elif 'walker' in p.env.name:

            # 'use normalEnv to be able to use extra info
            # env = gym.make('Walker2d-v3', reset_noise_scale = 0).unwrapped
            env = gym.make(
                'Walker2d-v3',
                terminate_when_unhealthy=terminate_when_unhealthy).unwrapped
            if SEED is None:
                env.unwrapped.seed(seed=p.agent.SEED)
            else:
                env.unwrapped.seed(seed=SEED)

            if p.env.prob_vel_penal is not None and p.env.prob_vel_penal > 0:
                dict_env = {'prob_vel_penal': p.env.prob_vel_penal,
                            'cost_vel': p.env.cost_vel,
                            'max_vel': p.env.max_vel}

                if terminate_when_unhealthy:
                    fname = f'walker2d_{expertise_level}_'\
                        f'redone_prob{dict_env["prob_vel_penal"]}'\
                        f'_penal{dict_env["cost_vel"]}_'\
                        f'maxvel{dict_env["max_vel"]}.hdf5'
                else:
                    fname = f'walker2d_{expertise_level}_'\
                        f'redone_prob{dict_env["prob_vel_penal"]}'\
                        f'_penal{dict_env["cost_vel"]}_'\
                        f'maxvel{dict_env["max_vel"]}_noterminate.hdf5'

                env = RewardHighVelocity(env, **dict_env)

            elif p.env.prob_unhealthy_penal is not None and \
                    p.env.prob_unhealthy_penal > 0:
                dict_env = {'prob_unhealthy_penal': p.env.prob_unhealthy_penal,
                            'cost_pose': p.env.cost_pose,
                            }

                if terminate_when_unhealthy:
                    fname = f'walker2d_{expertise_level}_'\
                        f'redone_prob{dict_env["prob_unhealthy_penal"]}'\
                        f'_penal{dict_env["cost_pose"]}_'\
                        f'unhealthy_position.hdf5'
                else:
                    fname = f'walker2d_{expertise_level}_'\
                        f'redone_prob{dict_env["prob_unhealthy_penal"]}'\
                        f'_penal{dict_env["cost_pose"]}_'\
                        'unhealthy_position_noterminate.hdf5'

                env = RewardUnhealthyPose(env, **dict_env)

            else:
                fname = f'walker2d_{expertise_level}_redone.hdf5'

            if not eval:

                h5path = os.path.join(
                    path, fname)
                dataset_file = h5py.File(h5path, 'r')
                data = {k: dataset_file[k][:] for k in get_keys(dataset_file)}
                dataset_file.close()
                print(f'Checking dataset {h5path} is correct')
                # Use get_dataset method from OfflineEnv Class in D4RL to
                # check dataset
                env_test = gym.make(p.env.name)
                env_test.get_dataset(h5path=h5path)

        elif 'hopper' in p.env.name:

            # use normalEnv to be able to use extra info
            env = gym.make(
                'Hopper-v3',
                terminate_when_unhealthy=terminate_when_unhealthy).unwrapped
            # env = gym.make('Hopper-v3', reset_noise_scale=0).unwrapped
            if SEED is None:
                env.unwrapped.seed(seed=p.agent.SEED)
            else:
                env.unwrapped.seed(seed=SEED)

            if p.env.prob_vel_penal is not None and p.env.prob_vel_penal > 0:
                dict_env = {'prob_vel_penal': p.env.prob_vel_penal,
                            'cost_vel': p.env.cost_vel,
                            'max_vel': p.env.max_vel}

                if terminate_when_unhealthy:
                    fname = f'hopper_{expertise_level}_'\
                        f'redone_prob{dict_env["prob_vel_penal"]}'\
                        f'_penal{dict_env["cost_vel"]}_'\
                        f'maxvel{dict_env["max_vel"]}.hdf5'
                else:
                    fname = f'hopper_{expertise_level}_'\
                        f'redone_prob{dict_env["prob_vel_penal"]}'\
                        f'_penal{dict_env["cost_vel"]}_'\
                        f'maxvel{dict_env["max_vel"]}_noterminate.hdf5'

                # use 'normal' env for
                env = RewardHighVelocity(env, **dict_env)
                # reproducibility (offlineEnv is not reproducible)

            elif p.env.prob_unhealthy_penal is not None and \
                    p.env.prob_unhealthy_penal > 0:
                dict_env = {'prob_unhealthy_penal': p.env.prob_unhealthy_penal,
                            'cost_pose': p.env.cost_pose,
                            }

                if terminate_when_unhealthy:
                    fname = f'hopper_{expertise_level}_'\
                        f'redone_prob{dict_env["prob_unhealthy_penal"]}'\
                        f'_penal{dict_env["cost_pose"]}_'\
                        f'unhealthy_position.hdf5'
                else:
                    fname = f'hopper_{expertise_level}_'\
                        f'redone_prob{dict_env["prob_unhealthy_penal"]}'\
                        f'_penal{dict_env["cost_pose"]}_'\
                        'unhealthy_position_noterminate.hdf5'

                env = RewardUnhealthyPose(env, **dict_env)

            else:
                fname = f'hopper_{expertise_level}_redone.hdf5'

            if not eval:
                h5path = os.path.join(
                    path, fname)
                dataset_file = h5py.File(h5path, 'r')
                data = {k: dataset_file[k][:] for k in get_keys(dataset_file)}
                dataset_file.close()
                print(f'Checking dataset {h5path} is correct')

                # Use get_dataset method from OfflineEnv Class in D4RL to
                # check dataset
                env_test = gym.make(p.env.name)
                env_test.get_dataset(h5path=h5path)

        else:
            raise ValueError(f'Environment {p.env.name} is not valid')

        # Prepare dataset for Offline Training:
        if not eval:
            dataset = D4RL_Dataset(data)
        if eval:
            dataset = None

        print(env, dataset)
        return env, dataset
