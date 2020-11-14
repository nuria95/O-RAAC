"""Utilities for the rllib library."""
import torch
__all__ = ['get_dict_hyperparams', 'compute_cvar']


def get_dict_hyperparams(p):
    hyper_params = {
        'batch_size': p.agent.BATCH_SIZE,
        'gamma': p.agent.GAMMA,
        'target_update_freq': p.agent.TARGET_UPDATE_FREQUENCY,
        'target_update_tau': p.agent.TARGET_UPDATE_TAU,
        'policy_update_freq': p.agent.POLICY_UPDATE_FREQUENCY,
        "lamda": p.agent.lamda,
        'n_quantiles_policy': p.agent.N_QUANTILES_POLICY,
        'n_quantiles_critic': p.agent.N_QUANTILES_CRITIC,
        "cvar": p.agent.cvar,
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
        raise ValueError(f'Not enough samples to compute {alpha} '
                         f'CVaR from {data}')
        # return sorted_data[:, 0].numpy()

    else:
        return cvar.numpy()
