"""Python Script Template."""
from .DDPG import DDPG
import torch
import torch.nn as nn


def rbf(x, y, sigma):
    """Compute RBF kernel."""
    if x.ndim == 3:
        b1, n, k1 = x.shape
        b2, m, k2 = y.shape
        diff = torch.square(x.repeat(1, m, 1) -
                            y.repeat_interleave(n, 1)).sum(-1)
        return torch.exp(-diff / sigma).reshape(b1, n, m)
    else:
        diff = torch.square(x - y).sum(-1)
        return torch.exp(-diff / sigma)


def laplace(x, y, sigma):
    """Compute Laplace kernel."""
    if x.ndim == 3:
        b1, n, k1 = x.shape
        b2, m, k2 = y.shape
        diff = torch.abs(x.repeat(1, m, 1) - y.repeat_interleave(n, 1)).sum(-1)
        return torch.exp(-diff / sigma).reshape(b1, n, m)
    else:
        diff = torch.abs(x - y).sum(-1)
        return torch.exp(-diff / sigma)


class MMDLoss(nn.Module):
    def __init__(self, kernel_function, kernel_param, epsilon,
                 regularization=False):
        super().__init__()
        self.sigma = kernel_param
        if kernel_function == "rbf":
            self.kernel_function = rbf
        elif kernel_function == "laplace":
            self.kernel_function = laplace
        else:
            raise NotImplementedError

        self.regularization = regularization
        if self.regularization:
            if not isinstance(epsilon, torch.Tensor):
                epsilon = torch.tensor(epsilon).float()
            self.dual_raw = torch.log(torch.exp(epsilon) - 1.0)
            self.epsilon = 0
        else:  # Constraint
            x = torch.tensor(1.0).float()
            self.dual_raw = torch.log(torch.exp(x) - 1.0)
            self.dual_raw.requires_grad = True
            self.epsilon = epsilon

    @property
    def dual(self):
        """Get dual variable"""
        return torch.nn.functional.softplus(self.dual_raw) + 1e-6

    def forward(self, behavior_actions, proposed_actions):
        mmd = self.mmd(x=behavior_actions, y=proposed_actions)
        if self.regularization:
            return self.dual * mmd
        else:
            return self.dual * (self.epsilon - mmd.detach()) + \
                self.dual.detach() * mmd

    def mmd(self, x, y):
        """Compute MMD loss"""
        return (
            self.kernel_function(x, x, self.sigma).mean()
            - 2 * self.kernel_function(x, y, self.sigma).mean()
            + self.kernel_function(y, y, self.sigma).mean()
        )


class BEAR(DDPG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_ = 0.4
        self.mmd_loss = MMDLoss(
            kernel_function=self.hyper_params.get("kernel", "rbf"),
            kernel_param=self.hyper_params.get("sigma", 20.),
            epsilon=self.hyper_params.get("epsilon",  0.05),
            regularization=self.hyper_params.get("regularization", False)
        )
        self.mmd_samples = self.hyper_params.get("mmd_samples", 5)
        self.optimizer_dual = torch.optim.Adam(
            params=[self.mmd_loss.dual_raw], lr=0.1)

    def train_actor(self, obs):
        """Train actor."""
        # Compute actor loss
        state, action, reward, done, next_state = obs

        policy_actions = self.policy(state)

        actor_loss = self.critic(state, policy_actions)
        if self.critic.num_heads > 1:
            actor_loss = actor_loss.min(-1)[0]
        actor_loss = -actor_loss.mean()

        actor_loss += self.mmd_loss(
            behavior_actions=action.unsqueeze(1),
            proposed_actions=policy_actions.unsqueeze(1).repeat_interleave(
                self.mmd_samples, 1
            )
        )

        # Optimize the actor and dual variables
        self.optimizer_actor.zero_grad()
        self.optimizer_dual.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        self.optimizer_dual.step()

    @property
    def model_dict(self):
        return {
            'critic': self.critic.state_dict(),
            'actor': self.policy.state_dict()
        }
