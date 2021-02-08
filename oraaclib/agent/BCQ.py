"""Python Script Template."""
from .DDPG import DDPG
import torch
import copy
from oraaclib.util.neural_networks.neural_networks import VAE, VAEActor


class BCQ(DDPG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dim_state = self.env.observation_space.shape[0]
        dim_action = self.env.action_space.shape[0]
        max_action = self.env.action_space.high[0]
        latent_dim = dim_action * 2

        self.policy = VAEActor(
            state_dim=dim_state,
            action_dim=dim_action,
            max_action=max_action,
            phi=self.hyper_params.get("phi", 0.05)
        )
        self.target_policy = copy.deepcopy(self.policy)
        self.optimizer_actor = torch.optim.Adam(
            params=self.policy.parameters(),
            # lr=self.hyper_params.get('lr_actor', 1e-3)
        )

        self.vae = VAE(
            dim_state=dim_state,
            dim_action=dim_action,
            latent_dim=latent_dim,
            max_action=max_action
        )
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters())

    def act(self, state):  # only for evaluation
        if self.only_vae:
            return self.act_vae(state)
        else:
            with torch.no_grad():
                state = torch.tensor(state.reshape(
                    1, -1)).float().repeat(100, 1)
                action = self.policy(state, self.vae.decode(state))
                ind = self.critic(state, action)[..., 0].argmax(0)
            return action[ind].cpu().data.numpy()

    def act_vae(self, state):  # only for evaluation
        with torch.no_grad():
            state = torch.tensor(state.reshape(1, -1)).float()
            vae_action = self.vae.decode(state)
        return vae_action.numpy()

    def train(self):
        super().train_step()  # count number of training steps

        obs = self.dataset.get_batch(batch_size=self.batch_size)
        self.train_vae(obs)
        self.train_critic(obs)

        if self.num_train_steps % self.policy_update_freq == 0:
            self.train_actor(obs)

            self.target_critic.params = self.critic.params
            self.target_policy.params = self.policy.params

    def train_vae(self, obs):
        state, action, next_state, reward, not_done = obs

        # Variational Auto-Encoder Training
        recon, mean, std = self.vae(state, action)
        recon_loss = torch.nn.functional.mse_loss(recon, action)
        kl_loss = -0.5 * (1 + torch.log(std.pow(2)) -
                          mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * kl_loss

        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

    def train_critic(self, obs):
        """Train critic."""
        state, action, reward, done, next_state = obs
        # Compute the target Q value
        with torch.no_grad():
            # Duplicate next state 10 times
            next_state = torch.repeat_interleave(next_state, 10, 0)

            # Compute value of perturbed actions sampled from the VAE
            target_q = self.target_critic(
                next_state,
                self.target_policy(next_state, self.vae.decode(next_state))
            )
            target_q = self._get_target(reward, done, target_q)

        # Get current Q estimate
        current_q = self.critic(state, action)

        # Compute critic loss
        critic_loss = torch.nn.functional.mse_loss(current_q, target_q)

        # Optimize the critic
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

    def train_actor(self, obs):
        """Train actor."""
        # Compute actor loss
        state, action, reward, done, next_state = obs

        # Pertubation Model / Action Training
        sampled_actions = self.vae.decode(state)
        perturbed_actions = self.policy(state, sampled_actions)

        # Update through DPG
        actor_loss = -self.critic(state, perturbed_actions)
        if self.critic.num_heads > 1:
            actor_loss = actor_loss[..., 0]
        actor_loss = actor_loss.mean()

        # Optimize the actor
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

    @property
    def model_dict(self):
        return {
            'critic': self.critic.state_dict(),
            'actor': self.policy.state_dict(),
            'vae': self.vae.state_dict()
        }
