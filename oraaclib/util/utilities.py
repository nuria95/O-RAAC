"""Utilities for the oraaclib library."""
import numpy as np
import torch
from torch.distributions import uniform
from torch.distributions.normal import Normal
import warnings

__all__ = ['get_dict_hyperparams', 'compute_cvar', 'Wang_distortion']


def get_dict_hyperparams(p):
    hyper_params = {
        'batch_size': p.agent.BATCH_SIZE,
        'gamma': p.agent.GAMMA,
        'target_update_freq': p.agent.TARGET_UPDATE_FREQUENCY,
        'target_update_tau': p.agent.TARGET_UPDATE_TAU,
        'policy_update_freq': p.agent.POLICY_UPDATE_FREQUENCY,
        "lamda": p.agent.lamda,
        "phi": p.agent.phi,
        'n_quantiles_policy': p.agent.N_QUANTILES_POLICY,
        'n_quantiles_critic': p.agent.N_QUANTILES_CRITIC,
        'risk_distortion': p.agent.RISK_DISTORTION,
        "alpha_cvar": p.agent.alpha_cvar,
        'max_eval_steps': p.agent.MAX_EVAL_STEPS,
        "lr_critic": p.agent.LEARNING_RATE_CRITIC,
        "lr_actor": p.agent.LEARNING_RATE_ACTOR,

        'memory_max_size': p.agent.MEMORY_MAX_SIZE,
        'num_workers': p.agent.NUM_WORKERS,
        'action_random_prob': p.agent.action_random_prob,
        "exploration_type": p.agent.exploration_type,
        "noise_eps_start": p.agent.noise_eps_start,
        "noise_eps_end": p.agent.noise_eps_end,
        "noise_eps_decay": p.agent.noise_eps_decay,

    }

    return hyper_params


def compute_cvar(data, alpha):
    if not isinstance(data, torch.Tensor):
        data = torch.Tensor(data)
    if len(data.size()) < 2:
        data.unsqueeze_(0)
    batch_size, N = data.size()
    sorted_data, _ = torch.sort(data)

    if alpha == 1 or alpha <= 0.5:
        cvar = sorted_data[:, :int(alpha * N)].mean(1)
    else:
        cvar = sorted_data[:, int(alpha * N)::].mean(1)
    if all(torch.isnan(cvar)):
        # raise ValueError(f'Not enough samples to compute {alpha} '
        #                  f'CVaR from {data}')
        warnings.warn(f'\nNot enough samples (N) to compute {alpha}-'
                      f'CVaR from tensor of episode cumrewards: {data}.'
                      '\nIncrease number of evaluation episodes N by passing'
                      'the argument --numexp N')
        return data[0].numpy()

    else:
        return cvar.numpy()


class Wang_distortion():
    """Sample quantile levels for the Wang risk measure.
    Wang 2000

    Parameters
    ----------
    eta: float. Default: -0.75
        for eta < 0 prduces risk-averse.
    """

    def __init__(self, eta=-0.75):
        self.eta = eta
        self.normal = Normal(loc=torch.Tensor([0]), scale=torch.Tensor([1]))

    def sample(self, num_samples):
        """
        Parameters
        ----------
        num_samples: tuple. (num_samples,)

        """
        taus_uniform = uniform.Uniform(0., 1.).sample(num_samples)
        wang_tau = self.normal.cdf(
            value=self.normal.icdf(value=taus_uniform) + self.eta)
        return wang_tau


class CPW(object):
    """Sample quantile levels for the CPW risk measure.

    Parameters
    ----------
    eta: float.
    """

    def __init__(self, eta=0.71):
        self.eta = eta

    def sample(self, num_samples):
        """
        Parameters
        ----------
        num_samples: tuple. (num_samples,)

        """
        taus_uniform = uniform.Uniform(0., 1.).sample(num_samples)
        tau_eta = taus_uniform ** self.eta
        one_tau_eta = (1 - taus_uniform) ** self.eta
        cpw_tau = tau_eta / ((tau_eta + one_tau_eta) ** (1. / self.eta))

        return cpw_tau


class Power(object):
    """Sample quantile levels for the Power risk measure.

    Parameters
    ----------
    eta: float. if eta < 0 is risk averse, if eta > 0 is risk seeking.
    """

    def __init__(self, eta=-2):
        self.eta = eta
        self.exponent = 1 / (1 + np.abs(eta))

    def sample(self, num_samples):
        """
        Parameters
        ----------
        num_samples: tuple. (num_samples,)

        """
        taus_uniform = uniform.Uniform(0., 1.).sample(num_samples)

        if self.eta > 0:
            return taus_uniform ** self.exponent
        else:
            return 1 - (1 - taus_uniform) ** self.exponent
