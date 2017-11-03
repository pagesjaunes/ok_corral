# coding=utf-8

import numpy as np
import random
import abc
import csv
import os

from ok_corral.agents.agents import ContextualBandit

HORIZON_PAR_DEFAUT = 100000

NOMBRE_BRAS_PAR_DEFAUT = 100
NOMBRE_LOG_PAR_DEFAUT = 100


class ContextualBanditEnvironment():
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.cumulativeReward = 0
        self._initialization_environment()


    @abc.abstractmethod
    def get_actions_description(self):
        pass

    @abc.abstractmethod
    def get_environment_description(self):
        pass

    @abc.abstractmethod
    def _initialization_environment(self):
        pass

    @abc.abstractmethod
    def _get_next_context(self):
        pass

    def run(self, p_agent, p_horizon=HORIZON_PAR_DEFAUT):
        """
        :type p_agent: ContextualBandit
        :param p_agent:
        :param p_horizon:
        :return:
        """

        cumulative_reward = 0
        cumulative_regret = 0

        cumulative_rewards = []
        cumulative_regrets = []

        for t in range(p_horizon):

            if t % (p_horizon / NOMBRE_LOG_PAR_DEFAUT) == 0:
                cumulative_rewards.append([t, cumulative_reward])
                cumulative_regrets.append([t, cumulative_regret])

            reward, regret = self._play(p_agent)
            cumulative_reward += reward
            cumulative_regret += regret

        cumulative_rewards.append([t + 1, cumulative_reward])
        cumulative_regrets.append([t + 1, cumulative_regret])

        return cumulative_rewards, cumulative_regrets

    @abc.abstractmethod
    def _perform_action(self, p_action):
        return

    def _play(self, p_agent):

        context = self._get_next_context()
        action = p_agent.select_action(context)

        reward, regret = self._perform_action(action)
        p_agent.observe(context, action, reward)

        return reward, regret



class Adult(ContextualBanditEnvironment):


    def _initialization_environment(self):

        self.nombre_bras = 14
        self.dimension = 82
        self.t = 0

        data = []

        with open('test_cases/environments/data/Adult.txt', 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                data.append(row)

        data = np.array(data, np.float32)

        self.features = data[:, :82]
        self.rewards = data[:, 82:]

    def get_actions_description(self):

        return self.nombre_bras

    def get_environment_description(self):

        return self.dimension

    def _perform_action(self, p_action):

        reward = self.rewards[self.t % len(self.rewards)][p_action]
        self.t += 1

        return reward, abs(1.-reward)

    def _get_next_context(self):
        return np.reshape(self.features[self.t % len(self.rewards)],(82,1))