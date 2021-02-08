import numpy as np
import torch

DEVICE = torch.device('cpu')

__all__ = ['EarlyStopping']


class EarlyStopping:
    """Early stops training if param doesn't improve after a given patience.
    """

    def __init__(self, name_save, patience=7, verbose=False, delta=0,
                 evol_type='loss', agent_properties=None, env_properties=None):
        """

        Args:
            patience (int): How long to wait after last time validation loss
                            improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss
                            improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify
                            as an improvement. Default: 0
        """

        self.patience = patience
        self.verbose = verbose
        self.evol_type = evol_type
        if name_save is not None:
            self.loss_evol_file = f'/{self.evol_type}_evol_' + \
                name_save.split('/')[-1]+'.json'
            self.loss_evol_file = '/'.join(name_save.split('/')
                                           [:-1])+self.loss_evol_file
            with open(self.loss_evol_file, "w") as f:
                f.write('')
        self.counter = 0
        self.best_score = None
        self.best_mean = -np.Inf
        self.early_stop = False
        self.loss_min = np.Inf
        self.mean_max = -np.Inf
        self.cvar_max = -np.Inf
        self.delta = delta
        self.name_save = name_save
        self.agent_properties = agent_properties
        self.env_properties = env_properties

    def call_loss(self, critic_loss, actor_loss, episode_num, model_dict):

        score = -(critic_loss + actor_loss)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(-score, episode_num, model_dict)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(-score, episode_num, model_dict)
            self.counter = 0

    def call_mean(self, score, episode_num, model_dict):

        if score <= self.best_mean:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint_mean(score, episode_num, model_dict)
            self.counter = 0

    def call_cvar_mean(self, mean, cvar, episode_num, model_dict):

        if cvar > self.cvar_max:
            self.save_checkpoint_var(cvar, mean, episode_num, model_dict)

        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, loss, episode_num, model_dict):
        """Save model when validation loss/score decrease/incresase."""

        if self.verbose:
            print(
                f'Validation {self.evol_type} changed ({self.loss_min:.2f} -->'
                f'{loss:.2f}).  Saving model ...')

        with open(self.loss_evol_file, 'a') as f:
            f.write(
                f'Validation {self.evol_type} changed({self.loss_min: .2f}'
                f' - -> {loss: .2f}).'
                f'Saving model of episode {episode_num}\n')

        model_dict['env_properties'] = [i for i in self.env_properties.items()]
        model_dict['agent_properties'] = [
            i for i in self.agent_properties.items()]
        directory_dict = '{}_{}.tar'.format(self.name_save, self.evol_type)
        torch.save(model_dict, directory_dict)

        self.loss_min = loss

    def save_checkpoint_mean(self, mean, episode_num, model_dict):
        """Save model when validation loss/score decrease/incresase."""
        text_print = f'Saving improved {self.evol_type} model. '\
            f'Evaluation episode {episode_num}\n'\
            f'{self.evol_type} changed ({self.best_mean:.2f} -->'\
            f'{mean:.2f}).\n'
        if self.verbose:
            print(text_print)

        with open(self.loss_evol_file, 'a') as f:
            f.write(text_print)

        model_dict['env_properties'] = [i for i in self.env_properties.items()]
        model_dict['agent_properties'] = [
            i for i in self.agent_properties.items()]
        directory_dict = '{}_{}.tar'.format(self.name_save, self.evol_type)
        torch.save(model_dict, directory_dict)

        self.best_mean = mean

    def save_checkpoint_var(self, cvar, mean, episode_num, model_dict):
        """Save model when validation loss decrease."""

        text_print = f'Saving improved {self.evol_type} model. '\
            f'Evaluation episode {episode_num}\n'\
            f'{self.evol_type} changed ({self.cvar_max:.2f} -->'\
            f'{cvar:.2f}). '\
            f'For this evaluation mean is {mean:.2f}.\n '

        if self.verbose:
            print(text_print)

        with open(self.loss_evol_file, 'a') as f:
            f.write(text_print)

        model_dict['env_properties'] = [i for i in self.env_properties.items()]
        model_dict['agent_properties'] = [
            i for i in self.agent_properties.items()]
        directory_dict = '{}_{}.tar'.format(self.name_save, self.evol_type)

        torch.save(model_dict, directory_dict)
        self.mean_max = mean
        self.cvar_max = cvar
