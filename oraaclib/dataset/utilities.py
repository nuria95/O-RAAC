"""Tree for sampling Prioritzed Experience Replay."""
import numpy as np
import torch
from oraaclib.dataset import Observation
import h5py


def _cast_to_iter_class(generator, class_):
    if class_ in (tuple, list):
        return class_(generator)
    else:
        return class_(*generator)


def stack_list_of_tuples(iter_):
    """Convert a list of observation tuples to a list of numpy arrays.
    Parameters
    ----------
    iter_: list
        Each entry represents one row in the resulting vectors.
    Returns
    -------
    *arrays
        One stacked array for each entry in the tuple.

    References
    ----------
    Code from https://github.com/sebascuri/rllib.git.
    """
    try:
        generator = map(torch.stack, zip(*iter_))
        return _cast_to_iter_class(generator, iter_[0].__class__)
    except TypeError:
        generator = map(np.stack, zip(*iter_))
        return _cast_to_iter_class(generator, iter_[0].__class__)


class SumTree():
    """A Binary Tree data structure.

    Creates a binary tree data structure with 'max_len' children leaves and
    with parent's  being sum of its childrens.
    Length of tree is then = (2*max_len) -1.
    Leaf nodes store the transition priorities and the internal nodes are
    intermediate sums, with the parent node containing the sum over all
    priorities, total_cumsum.

    Parameters
    ----------
    max_len: int
        buffer size of experience replay algorithm.
    """

    write = 0

    def __init__(self, max_len):
        self.max_len = max_len
        self.tree = np.zeros(2 * max_len - 1)
        self.data = np.zeros(max_len, dtype=object)
        self.n_entries = 0

    def _retrieve(self, idx, s):
        """Find sample on leaf node.

        Idea is to start with idx 0, root node, and keep comparing the s value
        with the  root values. Function keeps calling itself, while going down
        the tree, comparing with lower and lower parent nodes in the tree,
        till reaching the first conditional in which the index of the tree
        is already in the leaf layer.

        """
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):  #  reached leaf layer (where priorities
            # values are)
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)  # keep in left part of tree
        else:
            # move to right part
            return self._retrieve(right, s - self.tree[left])

    def total_cumsum(self):
        """Return total cumulative sum: root vaulue."""
        return self.tree[0]

    def add(self, p, observation):
        """Store priority and sample."""
        if not type(observation) == Observation:
            raise TypeError("""
            input has to be of type Observation, and it was found of type {}
            """.format(type(observation)))

        # idx: idx of children in tree {cap-1,...,2*(cap)-2}
        idx = self.write + self.max_len - 1

        self.data[self.write] = observation
        self.update(idx, p)

        # write: idx of Data array from {0,...,cap-1}
        self.write += 1
        if self.write >= self.max_len:
            self.write = 0

        if not self.is_full:
            self.n_entries += 1

    def update(self, idx, p):
        """Update priority comparing to previous value in tree."""
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def _propagate(self, idx, change):
        """Propagate tree cumsum values from leaves to root."""
        parent = (idx - 1) // 2

        self.tree[parent] += change
        # update till reach root
        if parent != 0:
            self._propagate(parent, change)

    def get(self, s):
        """Get idx in tree, priority and sample observation."""
        idx = self._retrieve(0, s)  # idx on tree for priority value
        dataIdx = idx - self.max_len + 1  # idx on Data for observation

        return (idx, self.tree[idx], self.data[dataIdx])

    @property
    def is_full(self):
        """Flag that checks if SumTree in buffer is full.

        Returns
        -------
        bool
        """
        return self.n_entries >= self.max_len


class DatasetWriter(object):
    def __init__(self, mujoco=False, goal=False):
        self.mujoco = mujoco
        self.goal = goal
        self.data = self._reset_data()
        self._num_samples = 0

    def _reset_data(self):
        data = {'observations': [],
                'actions': [],
                'terminals': [],
                'rewards': [],
                }
        if self.mujoco:
            data['infos/qpos'] = []
            data['infos/qvel'] = []
            data['infos/reward_ctrl'] = []
            data['infos/reward_run'] = []
            data['infos/x_vel'] = []
            data['infos/angle'] = []
        if self.goal:
            data['infos/goal'] = []
        return data

    def __len__(self):
        return self._num_samples

    def append_data(self, s, a, r, done, goal=None, mujoco_env_data=None,
                    info=None):
        self._num_samples += 1
        self.data['observations'].append(s)
        self.data['actions'].append(a)
        self.data['rewards'].append(r)
        self.data['terminals'].append(done)
        if self.goal:
            self.data['infos/goal'].append(goal)
        if self.mujoco and mujoco_env_data is not None:
            self.data['infos/qpos'].append(mujoco_env_data[0])  # qpos
            self.data['infos/qvel'].append(mujoco_env_data[1])  # qvel
            self.data['infos/x_vel'].append(info['x_velocity'])
            try:
                self.data['infos/angle'].append(info['angle'])
            except KeyError:
                pass

    def write_dataset(self, fname, max_size=None, compression='gzip'):
        np_data = {}
        for k in self.data:
            if k == 'terminals':
                dtype = np.bool_
            else:
                dtype = np.float32
            data = np.array(self.data[k], dtype=dtype)
            if max_size is not None:
                data = data[:max_size]
            np_data[k] = data

        dataset = h5py.File(fname, 'w')
        for k in np_data:
            dataset.create_dataset(k, data=np_data[k], compression=compression)
        dataset.close()
