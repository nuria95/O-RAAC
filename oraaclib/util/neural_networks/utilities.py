import torch.nn as nn
__all__ = ['parse_nonlinearity', 'parse_layers', 'update_parameters']


def parse_nonlinearity(non_linearity):
    """Parse non-linearity.
    References
    ----------
    Code from https://github.com/sebascuri/rllib.git."""

    if hasattr(nn, non_linearity):
        return getattr(nn, non_linearity)
    elif hasattr(nn, non_linearity.capitalize()):
        return getattr(nn, non_linearity.capitalize())
    elif hasattr(nn, non_linearity.upper()):
        return getattr(nn, non_linearity.upper())
    else:
        raise NotImplementedError(
            f"non-linearity {non_linearity} not implemented")


def parse_layers(layers, in_dim, non_linearity, normalized=False):
    """Parse layers of nn.
    References
    ----------
    Code from https://github.com/sebascuri/rllib.git."""
    if layers is None:
        layers = []
    elif isinstance(layers, int):
        layers = [layers]

    nonlinearity = parse_nonlinearity(non_linearity)
    layers_ = list()
    for layer in layers:
        layers_.append(nn.Linear(in_dim, layer))
        if normalized:
            layers_[-1].weight.data.normal_(0, 0.1)
        layers_.append(nonlinearity())
        in_dim = layer

    return nn.Sequential(*layers_), in_dim


def update_parameters(target_params, new_params, tau=1.0):
    """Update the parameters of target_params by those of new_params (softly).

    The parameters of target_nn are replaced by:
        target_params <- (1-tau) * (target_params) + tau * (new_params)

    Parameters
    ----------
    target_params: iter
    new_params: iter
    tau: float, optional

    Returns
    -------
    None.

    References
    ----------
    Code from https://github.com/sebascuri/rllib.git.
    """

    for target_param, new_param in zip(target_params, new_params):
        if target_param is new_param:
            continue
        else:
            new_param_ = ((1.0 - tau) * target_param.data.detach()
                          + tau * new_param.data.detach())
            target_param.data.copy_(new_param_)
