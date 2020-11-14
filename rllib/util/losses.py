import torch
import torch.nn.functional as F

__all__ = ['quantile_regression_loss', 'quantile_huber_loss']


def quantile_regression_loss(T_theta, Theta, tau_quantiles):
    """Compute quantile regression loss.

    Parameters
    ----------
    T_theta: torch.Tensor
            Target quantiles of size [batch_size x num_quantiles]

    Theta: torch.Tensor
            Current quantiles of size [batch_size x num_quantiles]
    tau_quantiles: torch.Tensor
        Quantile levles: [1xnum_quantiles]

    Returns
    -------
    loss: float
        Quantile regression loss
    """
    # Repeat Theta rows N times, amd stack batches in 3dim -->
    # -->[batch_size x N x N ]
    # (N = num quantiles)
    # Repeat T_Theta cols N times, amd stack batches in 3dim -->
    # --> [batch_size x N x N ]
    batch_size, num_quantiles = Theta.size()
    Theta_ = Theta.unsqueeze(2)  # batch_size, N, 1
    T_theta_ = T_theta.unsqueeze(1)  # batch_size, 1, N
    tau = tau_quantiles.unsqueeze(0).unsqueeze(2)  # 1, N,1
    error = T_theta_.expand(-1, num_quantiles, -1) - \
        Theta_.expand(-1, -1, num_quantiles)
    quantile_loss = torch.abs(tau - error.le(0.).float())  # (batch_size, N, N)
    loss_ = torch.mean(torch.mean(quantile_loss * error, dim=1).mean(dim=1))

    return loss_


def quantile_huber_loss(T_theta, Theta, tau_quantiles, k=1):
    """Compute quantile huber loss.

    Parameters
    ----------
    T_theta: torch.Tensor
            Target quantiles of size [batch_size x num_quantiles]

    Theta: torch.Tensor
            Current quantiles of size [batch_size x num_quantiles]
    tau_quantiles: torch.Tensor
        Quantile levles: [1xnum_quantiles]

    Returns
    -------
    loss: float
        Quantile Huber loss
    """
    # Repeat Theta rows N times, amd stack batches in 3dim -->
    # -->[batch_size x N x N ]
    # (N = num quantiles)
    # Repeat T_Theta cols N times, amd stack batches in 3dim -->
    # --> [batch_size x N x N ]

    batch_size, num_quantiles = Theta.size()
    Theta_ = Theta.unsqueeze(2)  # batch_size, N, 1
    T_theta_ = T_theta.unsqueeze(1)  # batch_size, 1, N
    tau = tau_quantiles.unsqueeze(0).unsqueeze(2)  # 1, N,1
    error = T_theta_ - Theta_  # all minus all [batch_size, N, N]

    quantile_loss = torch.abs(tau - error.le(0.).float())  # (batch_size, N, N)

    huber_loss_ = F.smooth_l1_loss(
        Theta_.expand(-1, -1, num_quantiles),
        T_theta_.expand(-1, num_quantiles, -1),
        reduction='none')

    loss_ = (quantile_loss * huber_loss_).mean()
    return loss_
