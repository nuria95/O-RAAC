"""Implementation of an Experience Replay Buffer."""
import random

import numpy as np
import torch
from torch.utils import data
from torch.utils.data._utils.collate import default_collate

from .datatypes import GAEObservation, Observation, WCPG_Observation
from .utilities import SumTree, stack_list_of_tuples


class ExperienceReplay(data.Dataset):
    """An Experience Replay Buffer dataset.

    The Experience Replay algorithm stores transitions and access them IID.
    It has a size, and it erases the older samples, once the buffer is full,
    like on a queue.

    Parameters
    ----------
    max_len: int
        buffer size of experience replay algorithm.
    transformations: list of transforms.AbstractTransform, optional
        A sequence of transformations to apply to the dataset, each of which
        is a callable that takes an observation as input and returns a modified
        observation. If they have an `update` method it will be called whenever
        a new trajectory is added to the dataset.

    Methods
    -------
    append(observation):
        append an observation to the dataset.
    shuffle():
        shuffle the dataset.
    is_full: bool
        check if buffer is full.

    References
    ----------
    Code modified from https://github.com/sebascuri/rllib.git.
    """

    def __init__(self, max_len, transformations: list = None):
        super().__init__()
        self._max_len = max_len
        # Observation is a namedtuple:
        # namedtuple('Observation',('state', 'action', 'reward',
        # 'next_state', 'done')))
        self._memory = np.empty((self._max_len,), dtype=Observation)
        self._sampling_idx = []
        self._ptr = 0
        self._transformations = transformations or list()

    def __getitem__(self, idx):
        """Return any desired observation.

        Parameters
        ----------
        idx: int

        Returns
        -------
        observation: Observation
        """
        # assert idx < len(self)
        idx = self._sampling_idx[idx]
        """This is done for shuffling. If memory was directly shuffled, then
        ordering would be lost.
        """

        observation = self._memory[idx]
        for transform in self._transformations:
            observation = transform(observation)
        weight = 1.0
        idx = np.random.randn()
        return observation, idx, weight

    def __len__(self):
        """Return the current size of the buffer."""
        if self.is_full:
            return self._max_len
        else:
            return self._ptr

    def append(self, observation):
        """Append new observation to the dataset.

        Parameters
        ----------
        observation: Observation

        Raises
        ------
        TypeError
            If the new observation is not of type Observation.
        """
        if not type(observation) == Observation:
            raise TypeError("""
            input has to be of type Observation, and it was found of type {}
            """.format(type(observation)))
        if not self.is_full:
            # array of indexes to sample from.
            # when is_full: _sampling_idx = [0,1....,max_len-1]
            self._sampling_idx.append(self._ptr)
        self._memory[self._ptr] = observation
        # _ptr, scalar ranging from 0 to [max_len-1],
        # where to substitute the memory obs.
        self._ptr = (self._ptr + 1) % self._max_len

        for transformation in self._transformations:
            transformation.update(observation)

    def shuffle(self):
        """Shuffle the dataset.

        We only shuffle the sampling_idx attribute, used for sampling batches
        when training. The _ptr attibute keeps unchanged so that we don't lose
        the ordering in the memory, and keep updating in a queue base.
        """
        np.random.shuffle(self._sampling_idx)  # only shuffle the sampling_idx

    @property
    def is_full(self):
        """Flag that checks if memory in buffer is full.

        Returns
        -------
        bool
        """
        return self._memory[-1] is not None  # check if the last element is not
        # empty.


# stored as ( s, a, r, s_ ) in SumTree
class Prioritized_ExperienceReplay(data.Dataset):
    """An Experience Replay Buffer dataset with stochastic proportional
    priorization.

    The Prioritized Experience Replay algorithm stores transitions in a tree
    and access them. It has a size, and it erases the older samples, once the
    buffer is full, like on a queue. It stores the TD error of every sample,
    and samples with higher "priority" samples with higher TD error.
    We ensure that the probability of being sampled is monotonic in a
    transition’s priority, while guaranteeing a non-zero
    probability even for the lowest-priority transitions.

    Parameters
    ----------
    max_len: int
        buffer size of experience replay algorithm.
    transformations: list of transforms.AbstractTransform, optional
        A sequence of transformations to apply to the dataset, each of which
        is a callable that takes an observation as input and returns a modified
        observation. If they have an `update` method it will be called whenever
        a new trajectory is added to the dataset.

    Methods
    -------
    append(observation):
        append an observation to the dataset.
    shuffle():
        shuffle the dataset.
    is_full: bool
        check if buffer is full.
    """

    epsilon = 0.01
    alpha = 0.2  # the smallest the less Prioritization and the more uniform
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, max_len,  transformations: list = None,
                 batch_size=None):
        super().__init__()
        self._max_len = max_len
        self._transformations = transformations or list()
        self.tree = SumTree(max_len)  # initialise tree
        self.batch_size = batch_size
        self._sampling_idx = []

    def __getitem__(self, idx):
        """Return any desired observation.

        Parameters
        ----------
        idx: int

        Returns
        -------
        observation: Observation
        """
        # assert idx < len(self)
        # idx = np.random.rand(0, len(self))  # make at random ?
        """Return observation, weight and index of tree.

        To sample a minibatch of size k, the range [0,total_error_cumsum] is
        divided equally into ranges. Next, a value is uniformly sampled
        from each range.
        Finally,transitions that correspond to each of these sampled values are
        retrieved from the tree.
        """
        segment_size = self.tree.total_cumsum() / self.batch_size
        # linear schedule for beta param: annealing the amount of
        # importance-sampling correction over time. The highest the more
        # bias correction
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        a = segment_size * idx
        b = segment_size * (idx + 1)
        s = random.uniform(a, b)
        (idx_tree, p, observation) = self.tree.get(s)

        sampling_probability = p / self.tree.total_cumsum()
        # Importance Sampling weights to reduce bias: estimate expected values
        # of reward under prioritized replay introduces bias --> correct
        # via IS weights on samples
        IS_weight = np.power(self.tree.n_entries *
                             sampling_probability, -self.beta)

        for transform in self._transformations:
            observation = transform(observation)
        return observation, idx_tree, IS_weight

    def __len__(self):
        """Return the current size of the buffer."""
        if self.is_full:
            return self._max_len
        else:
            return self.tree.n_entries

    def _get_priority(self, error):
        return(np.abs(error)+self.epsilon)**self.alpha

    def add_tree(self, error, observation):
        """Add an observation and corresponding error in the memory tree."""
        p = self._get_priority(error)
        self.tree.add(p, observation)

    def shuffle(self):
        """Shuffle the dataset."""
        np.random.shuffle(self._sampling_idx)

    def update(self, idx, error):
        """Update tree."""
        p = self._get_priority(error)
        self.tree.update(idx, p)

    @property
    def is_full(self):
        """Flag that checks if SumTree in buffer is full.

        Returns
        -------
        bool
        """
        return self.tree.n_entries >= self._max_len


class GAEExperienceReplay(data.Dataset):
    """An Experience Replay Buffer dataset to implement Generalized
    Advantage Estimation.

    The Experience Replay algorithm stores transitions and access them IID.
    It has a size, and it erases the older samples, once the buffer is full,
    like on a queue.

    Parameters
    ----------
    max_len: int
        buffer size of experience replay algorithm.
    transformations: list of transforms.AbstractTransform, optional
        A sequence of transformations to apply to the dataset, each of which
        is a callable that takes an observation as input and returns a modified
        observation. If they have an `update` method it will be called whenever
        a new trajectory is added to the dataset.

    Methods
    -------
    append(GAEobservation):
        append an GAEobservation to the dataset.
    shuffle():
        shuffle the dataset.
    is_full: bool
        check if buffer is full.

    References
    ----------
    John Schulman, Philipp Moritz,Sergey Levine,Michael Jordan,Pieter Abbeel
    High-Dimensional Continuous Control Using Generalized Advantage Estimation
    2016
    """

    def __init__(self, max_len, transformations: list = None):
        super().__init__()
        self._max_len = max_len
        # GAEObservation is a namedtuple:
        # namedtuple('GAEObservation',('state', 'action',
        # 'reward', 'next_state','done','gae_traj','gae_rew'))):
        self._memory = np.empty((self._max_len,), dtype=GAEObservation)
        self._sampling_idx = []
        self._ptr = 0
        self._transformations = transformations or list()

    def __getitem__(self, idx):
        """Return any desired observation.

        Parameters
        ----------
        idx: int

        Returns
        -------
        observation: GAEObservation

        """
        # assert idx < len(self)
        idx = self._sampling_idx[idx]
        """This is done for shuffling. If memory was directly shuffled,
        then ordering would be lost.
        """

        observation = self._memory[idx]
        for transform in self._transformations:
            observation = transform(observation)
        weight = 1.0
        idx = np.random.randn()
        return observation, idx, weight

    def __len__(self):
        """Return the current size of the buffer."""
        if self.is_full:
            return self._max_len
        else:
            return self._ptr

    def append(self, observation):
        """Append new observation to the dataset.

        Parameters
        ----------
        observation: Observation

        Raises
        ------
        TypeError
            If the new observation is not of type Observation.
        """
        if not type(observation) == GAEObservation:
            raise TypeError("""
            input has to be of type GAEObservation, and it was found of type {}
            """.format(type(observation)))
        if not self.is_full:
            # array of indexes to sample from.
            # when is_full: _sampling_idx = [0,1....,max_len-1]
            self._sampling_idx.append(self._ptr)
        self._memory[self._ptr] = observation
        # _ptr, scalar ranging from 0 to [max_len-1],
        # where to substitute the memory obs.
        self._ptr = (self._ptr + 1) % self._max_len

        for transformation in self._transformations:
            transformation.update(observation)

    def shuffle(self):
        """Shuffle the dataset.

        We only shuffle the sampling_idx attribute, used for sampling batches
        when training. The _ptr attibute keeps unchanged so that we don't
        lose the ordering in the memory, and keep updating in a queue base.
        """
        np.random.shuffle(self._sampling_idx)  # only shuffle the sampling_idx

    @property
    def is_full(self):
        """Flag that checks if memory in buffer is full.

        Returns
        -------
        bool
        """
        return self._memory[-1] is not None  # check if the last element is not
        # empty.


class D4RL_Dataset(data.Dataset):
    """An Off-policy dataset from D4RL datasets.

    Access transitions IID. It has a size, doesn't erase them.

    Parameters
    ----------
    dataset: obtained from OfflineEnv get_dataset method.

    Environments included in list of tasks in
    'https://github.com/rail-berkeley/d4rl/wiki/Tasks.'

    Methods
    -------
    get_batch: get a batch of data sampled iid.
    """

    def __init__(self, dataset):
        super().__init__()

        self.dataset = dataset
        self.keys = ['observations', 'actions', 'rewards', 'terminals']
        for key in self.keys:
            assert key in self.dataset, 'Dataset is missing key %s' % key

    def __len__(self):
        # (-1) so that don't raise
        return self.dataset['observations'].shape[0]-1
        # an error when asking for idx len() and appending next_state (len + 1)

    def __getitem__(self, idx):
        """Return any desired observation.

        Parameters
        ----------
        idx: int

        Returns
        -------
        observation: Observation

        """
        observation = [torch.as_tensor(
            self.dataset[k][idx], dtype=torch.float32) for k in self.keys]
        observation.append(torch.as_tensor(
            self.dataset['observations'][idx+1], dtype=torch.float32))
        return observation

    def get_batch(self, batch_size):
        """Get a batch of data sampled iid."""
        indeces = np.random.choice(len(self), batch_size)
        batch = self[indeces]

        return batch


class WCPG_ExperienceReplay(data.Dataset):
    """An Experience Replay Buffer dataset for WCPG algorithm.

    The Experience Replay algorithm stores transitions and access them IID.
    It has a size, and it erases the older samples, once the buffer is full,
    like on a queue.

    Parameters
    ----------
    max_len: int.
        buffer size of experience replay algorithm.
    transformations: list of transforms.AbstractTransform, optional.
        A sequence of transformations to apply to the dataset, each of which
        is a callable that takes an observation as input and returns a modified
        observation.
        If they have an `update` method it will be called whenever a new
        trajectory is added to the dataset.

    Methods
    -------
    append(observation) -> None:
        append an observation to the dataset.
    is_full: bool
        check if buffer is full.
    update(indexes, td_error):
        update experience replay sampling distribution with td_error feedback.
    all_data:
        Get all the transformed data.
    get_batch(batch_size):
        Get a batch of data.
    reset():
        Reset the memory to zero.
    get_observation(idx):
        Get the observation at a given index.

    References
    ----------
    Yichuan Charlie Tang and Jian Zhang and Ruslan Salakhutdinov
    Worst Cases Policy Gradients
    2019
    """

    def __init__(self, max_len, transformations=None, num_steps=1):
        super().__init__()
        self.max_len = max_len
        self.memory = np.empty((self.max_len,), dtype=WCPG_Observation)
        self.weights = torch.ones(self.max_len)
        self._ptr = 0
        self.transformations = transformations or list()
        self.num_steps = num_steps
        self.new_observation = True

    @classmethod
    def from_other(cls, other):
        """Create a Experience Replay from another one.

        All observations will be added sequentially, but only that will be
        copied.
        Weights will be initialized as if these were new observations.
        """
        new = cls(other.max_len, other.transformations, other.n_step)

        for observation in other.memory:
            if isinstance(observation, Observation):
                new.append(observation)
        return new

    def __len__(self):
        """Return the current size of the buffer."""
        if self.is_full:
            return self.max_len
        else:
            return self._ptr

    def __getitem__(self, idx):
        """Return any desired observation.

        Parameters
        ----------
        idx: int

        Returns
        -------
        observation: Observation
        idx: int
        weight: torch.tensor.

        """
        return self._get_observation(idx), idx, self.weights[idx]

    def _get_observation(self, idx):
        """Return any desired observation.

        Parameters
        ----------
        idx: int

        Returns
        -------
        observation: Observation
        """

        observation = self.memory[idx]
        for transform in self.transformations:
            observation = transform(observation)
        return observation

    def reset(self):
        """Reset memory to empty."""
        self.memory = np.empty((self.max_len,), dtype=Observation)
        self._ptr = 0

    def append(self, observation):
        """Append new observation to the dataset.

        Parameters
        ----------
        observation: Observation

        Raises
        ------
        TypeError
            If the new observation is not of type Observation.
        """
        if not type(observation) == WCPG_Observation:
            raise TypeError(
                f"input has to be of type Observation, and it was"
                f"{type(observation)}")

        self.memory[self._ptr] = observation
        self._ptr = (self._ptr + 1) % self.max_len

    def get_batch(self, batch_size):
        """Get a batch of data."""
        indices = np.random.choice(len(self), batch_size)
        return default_collate([self[i] for i in indices])

    @property
    def is_full(self):
        """Flag that checks if memory in buffer is full.

        Returns
        -------
        bool
        """
        return self.memory[-1] is not None  # check if the last element is
        # not empty.

    @property
    def all_data(self):
        """Get all the data."""
        all_obs = stack_list_of_tuples(self.memory[:self._ptr])

        for transformation in self.transformations:
            all_obs = transformation(all_obs)
        return all_obs

    def update(self, indexes, td_error):
        """Update experience replay sampling distribution with set of
        weights."""
        pass
