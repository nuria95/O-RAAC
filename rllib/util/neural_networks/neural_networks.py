"""Implementation of different Neural Networks with pytorch."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utilities import update_parameters, parse_layers

__all__ = ['VAE', 'ORAAC_Actor', 'RAAC_Actor', 'DeterministicNN_IQN',
           "Actor", "VAEActor", "Critic"
           ]


class VAE(nn.Module):
    """Vanilla Variational Auto-Encoder implementation.

    Parameters
    ----------
    dim_state: int
        dimension of state input to neural network.
    dim_action: int
        dimension of action input to neural network.
    latent_dim: int
        dimension of the latent vector
    max_action: Maxium value of action to scale to at the end

    References
    ----------
    Code based on Off-Policy Deep Reinforcement Learning without Exploration:
    https://github.com/sfujim/BCQ/blob/master/continuous_BCQ/BCQ.py
    """

    def __init__(self, dim_state, dim_action, latent_dim, max_action):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(dim_state + dim_action, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(dim_state + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, dim_action)

        self.max_action = max_action
        self.latent_dim = latent_dim

    def forward(self, state, action):
        """Execute forward computation of the VAE. Encode and Decode steps.
        Returns
        -------
        u: torch.Tensor
            [batch_size x dim_action] reconstructed action given latent vector.
        mean: torch.Tensor
            [batch_size x latent_dim] mean of the Gaussian distribution where
            latent vector is sampled from.
        std: torch.Tensor
            [batch_size x latent_dim] std f the Gaussian distribution where
            latent vector is sampled from.
        """
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        # Multivariate Gaussian with independent variables (covar is diagonal)
        # so we can sum the std directly:
        z = mean + std * torch.randn_like(std)  # batch_size x latent_dim
        u = self.decode(state, z)
        return u, mean, std

    def decode(self, state, z=None, eval=False):

        # When sampling from the VAE, latent vector is clipped to [-0.5, 0.5]
        if z is None:  # sample batch_size x latent_dim
            z = torch.randn((state.shape[0], self.latent_dim)).clamp(-0.5, 0.5)
        if eval is True:
            z = torch.zeros((state.shape[0], self.latent_dim))

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))


class ORAAC_Actor(nn.Module):
    """Risk-averse Policy for ORAAC (Perturbation model
    perturbing the risk-neutral action given by VAE)

    Parameters
    ----------
    dim_state: int
        dimension of state input to neural network.
    dim_action: int
        dimension of action input to neural network.
    max_action: Maximum value of action to scale to at the end
    lamda: float [0,1]
        percentage of perturbation added to the risk-neutral action
    tau: float [0,1], optional, default 1.0
        Regulates soft update of target parameters.
        % of new parameters used to update target parameters
    """

    def __init__(self, dim_state, dim_action, max_action, lamda=0.05, tau=1.0):
        super(ORAAC_Actor, self).__init__()
        self.l1 = nn.Linear(dim_state + dim_action, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, dim_action)

        self.max_action = max_action
        self.phi = nn.Parameter(torch.tensor(lamda), requires_grad=False)
        self.tau = tau

    def forward(self, state, action):
        """Execute forward computation of the Neural Network.

        Parameters
        ----------
        state: torch.Tensor
            Tensor of size [batch_size x dim_state]
        action: torch.Tensor
            Tensor of size [batch_size x dim_action]

        Returns
        -------
        a: torch.Tensor
            [batch_size x dim_action] (perturbed risk-averse action)
        """
        a = F.relu(self.l1(torch.cat([state, action], 1)))
        a = F.relu(self.l2(a))
        a = self.phi * self.max_action * torch.tanh(self.l3(a))
        return (a + action).clamp(-self.max_action, self.max_action)

    @property
    def params(self):  #  do not call it 'parameters' (already a default func)
        """Get iterator of NN parameters."""
        return self.parameters()

    @params.setter
    def params(self, new_params):
        """Set q-function parameters."""
        update_parameters(self.params, new_params, self.tau)


class RAAC_Actor(nn.Module):
    """Risk-averse Policy for RAAC (Not offline model so there is no
    risk-neutral action to perturb)

    Parameters
    ----------
    dim_state: int
        dimension of state input to neural network.
    dim_action: int
        dimension of action input to neural network.
    max_action: Maxium value of action to scale to at the end
    lamda: float [0,1]
        percentage of perturbation added to the risk-neutral action
    tau: float [0,1], optional, default 1.0
        Regulates soft update of target parameters.
        % of new parameters used to update target parameters
    """

    def __init__(self, dim_state, dim_action, max_action, lamda=0.05, tau=1.0):
        super(RAAC_Actor, self).__init__()
        self.l1 = nn.Linear(dim_state, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, dim_action)

        self.max_action = max_action
        self.tau = tau

    def forward(self, state):
        """Execute forward computation of the Neural Network.

        Parameters
        ----------
        state: torch.Tensor
            Tensor of size [batch_size x dim_state]
        Returns
        -------
        a: torch.Tensor
            [batch_size x dim_action] (risk-averse action)
        """
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.max_action * torch.tanh(self.l3(a))
        return a

    @property
    def params(self):  #  do not call it 'parameters' (already a default func)
        """Get iterator of NN parameters."""
        return self.parameters()

    @params.setter
    def params(self, new_params):
        """Set q-function parameters."""
        update_parameters(self.params, new_params, self.tau)


class DeterministicNN_IQN(nn.Module):
    """Deterministic NN Implementation for Implicit Quantile Network
    with continuous actions.
    Returns Q function given the triplet (state,action, tau) where tau
    is the confidence level.

    Parameters
    ----------
    dim_state: int
        dimension of state input to neural network.
    dim_action: int
        dimension of action input to neural network.
    layers_*: list of int, optional
        list of width of neural network layers, each separated with a
        'non_linearity' type non-linearity.
        *==state: layers mapping state input
        *==action: layers mapping action input
        *==f: layers mapping all 3 inputs together
    embedding_dim: dimension to map cat(state,action) to, and tau to.
    tau_embed_dim: int, optional, default 1
        if >1 map tau to a learned linear function of
        tau_embed_dim cosine basis functions of the form cos(pi*i*tau); where
        i = 1... tau_embed_dim. As in paper.

    biased_head: bool, optional, default = True
        flag that indicates if head of NN has a bias term or not.
    non_linearity: str, optional, default = 'ReLU'
        type of nonlinearity between layers

    tau: float [0,1], optional, default 1.0
        Regulates soft update of target parameters.
        % of new parameters used to update target parameters

     References
    ----------
    Will Dabney and Georg Ostrovski and David Silver and Rémi Munos
    Implicit Quantile Networks for Distributional Reinforcement Learning
    2018
    """

    def __init__(self, dim_state, dim_action,
                 layers_state: list = None,
                 layers_action: list = None,
                 layers_f: list = None,
                 embedding_dim=None,
                 tau_embed_dim=1,
                 biased_head=True,
                 non_linearity='ReLU',
                 tau=1.0):

        super().__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.layers_state = layers_state or list()
        self.layers_action = layers_action or list()
        self.layers_f = layers_f or list()
        self.embedding_dim = embedding_dim
        self.tau_embed_dim = tau_embed_dim
        self.tau = tau

        # Map state:
        self.fc_state, state_out_dim = parse_layers(
            layers_state, self.dim_state, non_linearity, normalized=True)
        # Map action:
        self.fc_action, action_out_dim = parse_layers(
            layers_action, self.dim_action, non_linearity, normalized=True)

        # Map cat(state,action) to embedding_dim
        # self.fc_state_action = nn.Sequential(layer_init(nn.Linear(
        # state_out_dim+action_out_dim,
        # self.embedding_dim, bias=biased_head), 1e-1), nn.ReLU())
        self.fc_state_action, _ = parse_layers(
            self.embedding_dim,
            state_out_dim + action_out_dim,
            non_linearity,
            normalized=True)

        # Prepare to map with cosine basis functions
        if self.tau_embed_dim > 1:
            self.i_ = torch.Tensor(np.arange(tau_embed_dim))

        # Map tau to embedding_dim
        self.head_tau, _ = parse_layers(self.embedding_dim,
                                        tau_embed_dim, non_linearity,
                                        normalized=True)
        # self.head_tau = nn.Sequential(
        #     layer_init(nn.Linear(tau_embed_dim, self.embedding_dim,
        #                          bias=biased_head), 1e-1), nn.ReLU())

        # Map [state,action,tau] to in_dim:
        self.hidden_layers_f, in_dim = parse_layers(
            layers_f, self.embedding_dim, non_linearity, normalized=True)
        # Layer mapping to 1-dim value function. No non-linearity added.
        self.head = nn.Linear(in_dim, 1, bias=biased_head)
        self.output_layer = nn.Sequential(self.hidden_layers_f, self.head)

        for name, param in self.named_parameters():
            if 'head.weight' in name:
                torch.nn.init.uniform_(param, -3e-4, 3e-4)
            if 'head.bias' in name:
                torch.nn.init.zeros_(param)

    def forward(self, state, tau_quantile, action=None):
        """Execute forward computation of the Neural Network.

        Parameters
        ----------
        state: torch.Tensor
            Tensor of size [batch_size x dim_state]
        action: torch.Tensor
            Tensor of size [batch_size x dim_action]
        tau_quantile: torch.Tensor
        Tensor of size [batch_size x 1]

        Returns
        -------
        output: torch.Tensor
            [batch_size x 1] (Q_function for triplet (state,action, tau)
        """

        state_output = self.fc_state(state)  # [batch_size x state_layer]
        action_output = self.fc_action(action)  # [batch_size x action_layer]
        state_action_output = self.fc_state_action(
            torch.cat((state_output, action_output), dim=-1))
        # [batch_size x  embedding_dim]

        # Cosine basis functions of the form cos(pi*i*tau)
        if self.tau_embed_dim > 1:
            a = torch.cos(torch.Tensor([math.pi])*self.i_*tau_quantile)
        else:
            a = tau_quantile
        tau_output = self.head_tau(a)  # [batch_size x embedding_dim]

        output = self.output_layer(
            torch.mul(state_action_output, tau_output)
        ).view(-1, 1)
        return output

    def get_sampled_Z(self, state, confidences, action):
        """Runs IQN for K different confidence levels
        Parameters
        ----------
        state: torch.Tensor [batch_size x dim_state]
        confidences: torch.Tensor. [1 x K]
        Returns
        -------
        Z_tau_K: torch.Tensor [batch_size x K]

        """
        K = confidences.size(0)  # number of confidence levels to evaluate
        batch_size = state.size(0) if state.dim() > 1 else 1
        # Reorganize so that the NN runs per one quantile at a time. Repeat
        # all batch_size block "num_quantiles" times:
        # [batch_size * K, dim_state]
        x = state.repeat(1, K).view(-1, self.dim_state)
        # [batch_size * K, dim_state]
        a = action.repeat(1, K).view(-1, self.dim_action)
        y = confidences.repeat(batch_size, 1).view(
            K*batch_size, 1)  # [batch_size * K, 1]
        Z_tau_K = self(state=x, tau_quantile=y, action=a).view(batch_size, K)
        return Z_tau_K

    @property
    def params(self):  #  do not call it 'parameters' (already a default func)
        """Get iterator of NN parameters."""
        return self.parameters()

    @params.setter
    def params(self, new_params):
        """Set parameters softly."""
        update_parameters(self.params, new_params, self.tau)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, tau=0.1):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action
        self.tau = tau

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

    @property
    def params(self):  #  do not call it 'parameters' (already a default func)
        """Get iterator of NN parameters."""
        return self.parameters()

    @params.setter
    def params(self, new_params):
        """Set q-function parameters."""
        update_parameters(self.params, new_params, self.tau)


class VAEActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, lamda=0.05, tau=0.1):
        super(VAEActor, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action
        self.lamda = lamda
        self.tau = tau

    def forward(self, state, action):
        a = F.relu(self.l1(torch.cat([state, action], 1)))
        a = F.relu(self.l2(a))
        a = self.lamda * self.max_action * torch.tanh(self.l3(a))
        return (a + action).clamp(-self.max_action, self.max_action)

    @property
    def params(self):  #  do not call it 'parameters' (already a default func)
        """Get iterator of NN parameters."""
        return self.parameters()

    @params.setter
    def params(self, new_params):
        """Set q-function parameters."""
        update_parameters(self.params, new_params, self.tau)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_heads=2, tau=0.005,
                 lambda_=0.75):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, num_heads)
        self.tau = tau
        self.lambda_ = lambda_
        self.num_heads = num_heads

    def forward(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        return self.l3(q)

    @property
    def params(self):  #  do not call it 'parameters' (already a default func)
        """Get iterator of NN parameters."""
        return self.parameters()

    @params.setter
    def params(self, new_params):
        """Set q-function parameters."""
        update_parameters(self.params, new_params, self.tau)
