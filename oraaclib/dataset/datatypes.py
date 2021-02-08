"""Project Data-types."""
from collections import namedtuple

import numpy as np


class Observation(namedtuple('Observation',
                             ('state', 'action', 'reward',
                              'next_state', 'done'))):
    """Observation datatype."""

    def __eq__(self, other):
        """Check if two observations are equal."""
        is_equal = np.allclose(self.state, other.state)
        is_equal &= np.allclose(self.action, other.action)
        is_equal &= np.allclose(self.reward, other.reward)
        is_equal &= np.allclose(self.next_state, other.next_state)
        is_equal &= self.done == other.done

        return is_equal

    def __ne__(self, other):
        """Check if two observations are not equal."""
        return not (self == other)


class GAEObservation(namedtuple('Observation',
                                ('state', 'action', 'reward', 'next_state',
                                 'done', 'state_traj', 'rew_traj',
                                 'done_traj'))):
    """Observation datatype for Generalized Advantage Estimation.

    References
    ----------
    John Schulman, Philipp Moritz,Sergey Levine,Michael Jordan,Pieter Abbeel
    High-Dimensional Continuous Control Using Generalized Advantage Estimation
    2016
    """

    def __eq__(self, other):
        """Check if two observations are equal."""
        is_equal = np.allclose(self.state, other.state)
        is_equal &= np.allclose(self.action, other.action)
        is_equal &= np.allclose(self.reward, other.reward)
        is_equal &= np.allclose(self.next_state, other.next_state)
        is_equal &= np.allclose(self.state_traj, other.state_traj)
        is_equal &= np.allclose(self.rew_traj, other.rew_traj)
        is_equal &= np.allclose(self.done_traj, other.done_traj)
        is_equal &= self.done == other.done

        return is_equal

    def __ne__(self, other):
        """Check if two observations are not equal."""
        return not (self == other)


class WCPG_Observation(namedtuple('Observation',
                                  ('state', 'action', 'reward', 'next_state',
                                   'done', 'alpha'))):
    """Observation datatype for Worst Case Policy Gradients.
    References
    ----------
    Yichuan Charlie Tang and Jian Zhang and Ruslan Salakhutdinov
    Worst Cases Policy Gradients
    2019
    """

    def __eq__(self, other):
        """Check if two observations are equal."""
        is_equal = np.allclose(self.state, other.state)
        is_equal &= np.allclose(self.action, other.action)
        is_equal &= np.allclose(self.reward, other.reward)
        is_equal &= np.allclose(self.next_state, other.next_state)
        is_equal &= np.allclose(self.alpha, other.alpha)
        is_equal &= self.done == other.done

        return is_equal

    def __ne__(self, other):
        """Check if two observations are not equal."""
        return not (self == other)
