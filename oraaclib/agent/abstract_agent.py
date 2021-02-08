"""Interface for agents."""
from abc import ABC, abstractmethod

import numpy as np
from oraaclib.util.utilities import compute_cvar


class AbstractAgent(ABC):
    """Interface for agents that interact with an environment.

    Parameters
    ----------

    Methods
    -------
    act(state): int or ndarray
        Given a state, it returns an action to input to the environment.
    observe(observation):
        Record an observation from the environment.
    start_episode:
        Start a new episode.
    end_episode:
        End an episode.
    end_interaction:
        End an interaction with an environment.
    policy:
        Return the policy the agent is using.

    References
    ----------
    Code modified from https://github.com/sebascuri/rllib.git.
    """

    def __init__(self):
        self._steps = {'total': 0, 'episode': 0}
        self._num_episodes = -1
        self.logs = {'total_steps': 0,
                     'episodes_steps': [],
                     'rewards': [],  # list of lists of rewards per step
                     'episodes_cumrewards': [],  # list of cumrewards of
                                                 # every episode
                     'episodes_td_errors': [],
                     'train_steps': -1,
                     'num_eval_episodes': -1,
                     'moving_av_reward': [],
                     'episodes_success': [],
                     'episodes_eval_success': [],
                     'moving_av_success': [],
                     'moving_av_eval_success_rate': [],
                     'eval_rewards': [],  # list of lists of rewards per step
                     'episodes_eval_steps': [],
                     'episodes_eval_cumrewards': [],  # list of cumrewards of
                                                      # every episode
                     'episode_cumvel': [],
                     'risky_state': [],
                     'episodes_vel': [],
                     'episodes_angles': [],
                     'episodes_vels': []}

    @abstractmethod
    def act(self, state):
        """Ask the agent for an action to interact with the environment.

        Parameters
        ----------
        state: int or ndarray

        Returns
        -------
        action: int or ndarray

        """
        raise NotImplementedError

    def start_episode(self):
        """Start a new episode."""
        self._num_episodes += 1

        self.logs['episodes_steps'].append(0)
        self.logs['episodes_cumrewards'].append(0)
        self.logs['rewards'].append([])

    def observe(self, observation):
        """Observe transition from the environment.

        Parameters
        ----------
        observation: Observation

        """
        self.logs['total_steps'] += 1
        self.logs['episodes_steps'][-1] += 1
        self.logs['episodes_cumrewards'][-1] += observation.reward
        # self.logs['rewards'][-1].append(observation.reward)

    def start_episode_offline(self, eval=False):
        """Start a new eval episode."""
        self.logs['num_eval_episodes'] += 1
        self.logs['episodes_eval_steps'].append(0)
        self.logs['episodes_eval_cumrewards'].append(0)
        # self.logs['eval_rewards'].append([])
        self.logs['risky_state'].append(0)
        self.logs['episode_cumvel'].append(0)
        if eval:
            self.logs['episodes_angles'].append([])
            self.logs['episodes_vels'].append([])

    def observe_offline(self, observation, info, eval=False):
        """Observe transition from the environment.

        Parameters
        ----------
        observation: Observation

        """
        self.logs['episodes_eval_steps'][-1] += 1
        self.logs['episodes_eval_cumrewards'][-1] += observation.reward
        self.logs['episode_cumvel'][-1] += round(info['x_velocity'], 2)
        if not observation.done:  # in case already reached terminal
            # state don't account
            try:
                self.logs['risky_state'][-1] += info['risky_state']*1
            except KeyError:
                pass
            try:
                self.logs['episode_cumvel'][-1] += round(info['x_velocity'], 2)
            except KeyError:
                pass
        if eval and 'angle' in info.keys():
            self.logs['episodes_angles'][-1].append(info['angle'])
        if eval and 'x_velocity' in info.keys():
            self.logs['episode_cumvel'][-1] += round(info['x_velocity'], 2)
            self.logs['episodes_vels'][-1].append(info['x_velocity'])

    def end_episode_offline(self):
        """End an episode."""

        self.logs['episodes_vel'].append(self.logs['episode_cumvel'][-1] /
                                         self.logs['episodes_eval_steps'][-1])

    def end_episode(self):
        """End an episode."""
        pass

    def end_interaction(self):
        """End the interaction with the environment."""
        pass

    def train_step(self):
        self.logs['train_steps'] += 1

    def eval_episode(self):
        self.logs['num_eval_episodes'] += 1

    def episode_success(self, success, num_av=5):
        self.logs['episodes_success'].append(success)
        if len(self.logs['episodes_success']) >= (num_av):
            ma = self.compute_moving_average_data(
                data=self.logs['episodes_success'],
                num_av=num_av)
            self.logs['moving_av_success'].append(ma)
        else:
            pass

    def compute_moving_average_data(self, data, num_av=10):
        if len(data) >= (num_av):
            return round((
                sum(data[-num_av:])/num_av), 2)
        else:
            pass

    @property
    def episodes_steps(self):
        """Return number of steps in current episode."""
        return self.logs['episodes_steps']

    @property
    # Num steps current episode
    def episode_steps(self):
        """Return number of steps in current episode."""
        return self.logs['episodes_steps'][-1]

    @property
    def episodes_rewards(self):
        """Return rewards in all the episodes seen."""
        return self.logs['rewards']

    @property
    def episodes_cumulative_rewards(self):
        """Return cumulative rewards in each episodes."""
        return self.logs['episodes_cumrewards']

    @property
    def episodes_eval_cumreward(self):
        """Return cumulative rewards in each eval episodes."""

        return self.logs['episodes_eval_cumrewards']

    @property
    def episodes_eval_rewards(self):
        """Return all rewards in every eval episode """
        return self.logs['eval_rewards']

    @property
    def episodes_eval_steps(self):
        """Return cumulative rewards in each eval episodes."""
        return self.logs['episodes_eval_steps']

    @property
    def moving_av_reward(self):
        """Return moving average vector reward.
        For some algs it is eval reward, for some it
        is every training reward."""
        return self.logs['moving_av_reward']

    @property
    def total_steps(self):
        """Return number of steps of interaction with environment."""
        return self.logs['total_steps']

    @property
    def num_eval_episodes(self):
        """Return number of evaluation episodes."""
        return self.logs['num_eval_episodes']

    @property
    def num_episodes(self):
        """Return number of episodes the agent
        interacted with the environment."""
        return self._num_episodes

    @property
    def num_train_steps(self):
        """Return number of training steps done."""
        return self.logs['train_steps']

    @property
    def episode_td_errors(self):
        """Return td error of last episode."""
        if len(self.logs['td_errors']) > 0:
            return sum(self.logs['td_errors'])/len(self.logs['td_errors'])
        else:
            return None

    @property
    def success_rates(self):
        return self.logs['episodes_success']

    def mean_eval_cumreward(self, times_eval):
        return np.mean(self.episodes_eval_cumreward[-times_eval:])

    def cvar_eval_cumreward(self, times_eval):
        return compute_cvar(
            self.episodes_eval_cumreward[-times_eval:], alpha=0.1)[0]

    def mean_ep_steps(self, times_eval):
        return np.mean(self.logs['episodes_eval_steps'][-times_eval:])

    def mean_vel_episodes(self, times_eval):
        return np.mean(self.logs['episodes_vel'][-times_eval:])

    def mean_risky_times(self, times_eval):
        return np.mean(self.logs['risky_state'][-times_eval:])

    def fraction_risky_times(self, times_eval):
        return np.mean(
            np.array(self.logs['risky_state'][-times_eval:])/np.array(
                self.logs['episodes_eval_steps'][-times_eval:]))
