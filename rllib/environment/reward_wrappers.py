import gym
from torch.distributions import Bernoulli

__all__ = ['RewardHighVelocity',
           'RewardUnhealthyPose',
           'RewardScale']


class RewardHighVelocity(gym.RewardWrapper):
    """Wrapper to modify environment rewards of 'Cheetah','Walker' and
    'Hopper'.

    Penalizes with certain probability if velocity of the agent is greater
    than a predefined max velocity.
    Parameters
    ----------
    kwargs: dict
    with keys:
    'prob_vel_penal': prob of penalization
    'cost_vel': cost of penalization
    'max_vel': max velocity

    Methods
    -------
    step(action): next_state, reward, done, info
    execute a step in the environment.
    """

    def __init__(self, env, **kwargs):
        super(RewardHighVelocity, self).__init__(env)
        self.penal_v_distr = Bernoulli(kwargs['prob_vel_penal'])
        self.penal = kwargs['cost_vel']
        self.max_vel = kwargs['max_vel']
        allowed_envs = ['Cheetah', 'Hopper', 'Walker']
        assert(any(e in self.env.unwrapped.spec.id for e in allowed_envs)), \
            'Env {self.env.unwrapped.spec.id} not allowed for RewardWrapper'

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        vel = info['x_velocity']
        info['risky_state'] = vel > self.max_vel
        info['angle'] = self.env.sim.data.qpos[2]

        if 'Cheetah' in self.env.unwrapped.spec.id:
            return (observation, self.new_reward(reward, info),
                    done, info)
        if 'Walker' in self.env.unwrapped.spec.id:
            return (observation, self.new_reward(reward, info),
                    done, info)
        if 'Hopper' in self.env.unwrapped.spec.id:
            return (observation, self.new_reward(reward, info),
                    done, info)

    def new_reward(self, reward, info):
        if 'Cheetah' in self.env.unwrapped.spec.id:
            forward_reward = info['reward_run']
        else:
            forward_reward = info['x_velocity']

        penal = info['risky_state'] * \
            self.penal_v_distr.sample().item() * self.penal

        # If penalty applied, substract the forward_reward from total_reward
        # original_reward = rew_healthy + forward_reward - cntrl_cost
        new_reward = penal + reward + (penal != 0) * (-forward_reward)
        return new_reward

    @property
    def name(self):
        return f'{self.__class__.__name__}{self.env}'


class RewardUnhealthyPose(gym.RewardWrapper):
    """Wrapper to modify environment rewards of 'Walker' and 'Hopper'.
    Penalizes with certain probability if pose of the agent doesn't lie
    in a 'robust' state space.
    Parameters
    ----------
    kwargs: dict
    with keys:
    'prob_pose_penal': prob of penalization
    'cost_pose': cost of penalization

    Methods
    -------
    step(action): next_state, reward, done, info
    execute a step in the environment.
    """

    def __init__(self, env, **kwargs):

        super(RewardUnhealthyPose, self).__init__(env)

        self.penal_distr = Bernoulli(kwargs['prob_pose_penal'])
        self.penal = kwargs['cost_pose']
        if 'Walker' in self.env.unwrapped.spec.id:
            self.robust_angle_range = (-0.5, 0.5)
            self.healthy_angle_range = (-1, 1)  # default env

        elif 'Hopper' in self.env.unwrapped.spec.id:
            self.robust_angle_range = (-0.1, 0.1)
            self.healthy_angle_range = (-0.2, 0.2)  # default env

        else:
            raise ValueError('Environment is not Walker neither Hopper '
                             f'for {self.__class__.__name__}')

    @property
    def is_robust_healthy(self):
        z, angle = self.env.sim.data.qpos[1:3]
        min_angle, max_angle = self.robust_angle_range
        robust_angle = min_angle < angle < max_angle
        is_robust_healthy = robust_angle  # and healthy_z
        return is_robust_healthy

    @property
    def is_healthy(self):
        z, angle = self.env.sim.data.qpos[1:3]
        h_min_angle, h_max_angle = self.healthy_angle_range
        healthy_angle = h_min_angle < angle < h_max_angle
        self.is_healthy = healthy_angle

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        info['risky_state'] = ~self.is_robust_healthy
        info['angle'] = self.env.sim.data.qpos[2]
        return observation, self.new_reward(reward), done, info

    def new_reward(self, reward):
        # Compute new reward according to penalty probability and agent state:

        # Penalty occurs if agent's pose is not robust with certain prob
        # If env.terminate when unhealthy=False (i.e. episode doesn't finish
        # when unhealthy pose), we do not add penalization when not in
        # healty pose.

        penal = (~self.is_robust_healthy) * (self.is_healthy) *\
            self.penal_distr.sample().item() * self.penal

        new_reward = penal + reward
        return new_reward

    @property
    def name(self):
        return f'{self.__class__.__name__}{self.env}'


class RewardScale(gym.RewardWrapper):
    def __init__(self, env, scale):

        gym.RewardWrapper.__init__(self, env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale
