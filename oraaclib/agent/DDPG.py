"""Python Script Template."""
from .abstract_agent import AbstractAgent
from oraaclib.util.neural_networks.neural_networks import Actor, Critic
from oraaclib.dataset import Observation

import torch
import copy


class DDPG(AbstractAgent):
    def __init__(self, env, hyper_params, dataset=None, early_stopper_rew=None,
                 early_stopper_var=None, logger=None, name_save=None,

                 eval=False, bool_save_model=True, only_vae=False):
        super().__init__()
        self.env = env
        self.dataset = dataset
        self.hyper_params = hyper_params
        self.eval = eval
        self.name_save = name_save
        self.early_stopper_var = early_stopper_var
        self.early_stopper_rew = early_stopper_rew
        self.bool_save_model = True
        self.logger = logger
        self.only_vae = only_vae

        dim_action = env.action_space.shape[0]
        dim_state = env.observation_space.shape[0]
        max_action = env.action_space.high[0]

        policy = Actor(
            state_dim=dim_state, action_dim=dim_action, max_action=max_action
        )
        self.policy = policy
        self.target_policy = copy.deepcopy(policy)

        critic = Critic(
            state_dim=dim_state,
            action_dim=dim_action,
            num_heads=hyper_params.get('num_heads', 2),
            tau=hyper_params.get('target_update_tau', 0.005),
            lambda_=hyper_params.get('lambda_', 0.75)
        )
        self.critic = critic
        self.target_critic = copy.deepcopy(critic)

        self.gamma = hyper_params.get('gamma', 0.99)
        self.policy_update_freq = hyper_params.get('target_update_freq', 2)
        self.batch_size = hyper_params.get('batch_size', 100)

        self.optimizer_actor = torch.optim.Adam(
            params=self.policy.parameters(),
            # lr=hyper_params.get('lr_actor', 1e-3)
        )
        self.optimizer_critic = torch.optim.Adam(
            params=self.critic.parameters(),
            # lr=hyper_params.get('lr_critic', 1e-3)
        )

    def save_model(self):
        if not self.bool_save_model:
            pass
        elif self.eval_episode > 30:
            self.early_stopper_rew.call_mean(
                score=self.mean_,
                episode_num=self.eval_episode,
                model_dict=self.model_dict)
            self.early_stopper_var.call_cvar_mean(
                mean=self.mean_,
                cvar=self.cvar_,
                episode_num=self.eval_episode,
                model_dict=self.model_dict)

    def save_final_model(self):
        if not self.bool_save_model:
            pass
        else:
            model_dict = self.model_dict
            self.logger.export_to_json()
            directory_dict = f'{self.name_save}_mean{self.mean_:.2f}_'\
                f'cvar{self.cvar_:.2f}epoch'\
                f'{self.eval_episode}.tar'
            torch.save(model_dict, directory_dict)
            print('Saving final model. End training')

    def act(self, state):  # only for evaluation
        with torch.no_grad():
            state = torch.tensor(state.reshape(1, -1)).float()
            action = self.policy(state)
        return action.data.numpy()

    def evaluate_model(self, max_episode_steps, times_eval=1, render=False):
        self.times_eval = times_eval
        with torch.no_grad():
            for i in range(times_eval):
                super().start_episode_offline(eval=self.eval)
                self.eval_episode = int(self.num_eval_episodes/times_eval)
                state = self.env.reset()
                done = False

                while not done:
                    if render:
                        self.env.render()
                    action = self.act(state)
                    next_state, reward, done, info = self.env.step(action)
                    observation = Observation(state=state,
                                              action=action,
                                              reward=reward,
                                              next_state=next_state,
                                              done=done)
                    self.observe_offline(observation, info, eval=self.eval)
                    state = next_state
                    if max_episode_steps <= self.episodes_eval_steps[-1]:
                        break
                super().end_episode_offline()
                print(f'Fraction Risky times:'
                    f'{self.fraction_risky_times(self.times_eval):.2f}\n\n')
        if self.logger:
            self.log_data()

    def train(self):
        super().train_step()  # count number of training steps

        obs = self.dataset.get_batch(batch_size=self.batch_size)
        self.train_critic(obs)

        if self.num_train_steps % self.policy_update_freq == 0:
            self.train_actor(obs)

            # call @params.setter in IQN NN
            self.target_critic.params = self.critic.params
            # call @params.setter in ORAAC NN
            self.target_policy.params = self.policy.params

    def _get_target(self, reward, done, target_q):
        if self.critic.num_heads > 1:
            q_min, q_max = target_q.min(-1)[0], target_q.max(-1)[0]
            lambda_ = self.critic.lambda_
            target_q = lambda_ * q_min + (1 - lambda_) * q_max
            target_q = target_q.reshape(
                self.batch_size, -1).max(1)[0].reshape(-1)

            target_q = (reward + (1 - done) * self.gamma * target_q).detach()

            target_q = target_q.unsqueeze(-1).repeat_interleave(
                self.critic.num_heads, -1
            )
        else:
            target_q = (reward + (1 - done) * self.gamma * target_q).detach()
        return target_q

    def train_critic(self, obs):
        """Train critic."""
        state, action, reward, done, next_state = obs
        # Compute the target Q value
        target_q = self.target_critic(
            next_state, self.target_policy(next_state))
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
        actor_loss = -self.critic(state, self.policy(state))
        if self.critic.num_heads > 1:
            actor_loss = actor_loss[..., 0]
        actor_loss = actor_loss.mean()

        # Optimize the actor
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

    def log_data(self):
        self.logger.add(
            **{"eval_mean_reward": self.mean_,
                "eval_cvar_reward": self.cvar_,
                "mean_ep_steps": self.mean_ep_steps(self.times_eval),
                "mean_vel_episodes": self.mean_vel_episodes(self.times_eval),
                "mean_risky_times": self.mean_risky_times(self.times_eval),
                "fraction_risky_times":
                    self.fraction_risky_times(self.times_eval)
               })
        if self.eval:
            self.logger.add(**{"angles": self.logs['episodes_angles']})
            self.logger.add(**{"velocities": self.logs['episodes_vels']})

        if not self.num_train_steps % 1000 or self.eval:
            self.logger.export_to_json()

    @property
    def mean_(self):
        mean = self.mean_eval_cumreward(self.times_eval)
        return mean

    @property
    def cvar_(self):
        cvar = self.cvar_eval_cumreward(self.times_eval)
        return cvar
