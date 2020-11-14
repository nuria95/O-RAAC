"""Implementation of a Logger class."""
import json
import os

import numpy as np
import torch


class Logger(object):
    """Class that implements a logger of statistics.

    Parameters
    ----------
    name: str

    References
    ----------
    Code from https://github.com/sebascuri/rllib.git.
    """

    def __init__(self, folder, name, comment=''):

        self.keys = []
        self.name = name
        self.folder = folder

        if not os.path.exists(folder):
            os.makedirs(folder)

    def add(self, **kwargs):
        """Add statistics in logger.

        Parameters
        ----------
        kwargs: dict
            Any kwargs passed to update is converted to numpy and averaged
            over the course of an episode.
        """
        for key, value in kwargs.items():
            if key not in self.keys:
                self.keys.append(key)
                # making 'key' new method of class = an empty listh
                setattr(self, key, [])
            if isinstance(value, torch.Tensor):
                value = value.detach().numpy()
            # value = np.nan_to_num(value) #was converting list to arrays :S
            if isinstance(value, np.float32):
                value = float(value)
            if isinstance(value, np.int64):
                value = int(value)
            # calling method key of class and since a list
            getattr(self, key).append(value)
            # appending new val

    # def placeholder(self):
    #     for k in self.keys:
    #         v = getattr(self, k)

    def export_to_json(self, hparams=None):
        """Save the statistics (and hparams) to a json file."""
        the_dict = vars(self)
        the_dict.pop('TimeLimit.truncated', None)
        with open(f"{self.folder}/{self.name}.json", "w") as f:
            json.dump(the_dict, f)
