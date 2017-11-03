# coding=utf-8
import numpy as np
import random
from ok_corral.agents.agents import Bandit
from scipy.stats import bernoulli

HORIZON_PAR_DEFAUT = 100000

NOMBRE_BRAS_PAR_DEFAUT = 100
NOMBRE_LOG_PAR_DEFAUT = 100


class BanditEnvironment():
    def __init__(self, p_nombre_bras=NOMBRE_BRAS_PAR_DEFAUT):
        self.nombre_bras = p_nombre_bras
        self.cumulativeReward = 0

    def get_actions_description(self):
        return self.nombre_bras

    def _initialization_environment(self):
        assert False, "initialization_environment doit être implémentée par la classe fille"

    def run(self, p_agent, p_horizon=HORIZON_PAR_DEFAUT):
        """
        :type p_agent: Bandit
        :param p_agent:
        :param p_horizon:
        :return:
        """

        cumulative_reward = 0
        cumulative_regret = 0

        cumulative_rewards = []
        cumulative_regrets = []

        self._initialization_environment()

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

    def _perform_action(self, p_action):
        assert False, "perform_action doit être implémentée par la classe fille"

        return None

    def _play(self, p_agent):
        action = p_agent.select_action()
        reward, regret = self._perform_action(action)
        p_agent.observe(action, reward)

        return reward, regret


DELTA_PAR_DEFAUT = 0.01


class StochasticBanditEnvironment(BanditEnvironment):

    def __init__(self, p_nombre_bras = NOMBRE_BRAS_PAR_DEFAUT, p_delta=DELTA_PAR_DEFAUT):
        BanditEnvironment.__init__(self, p_nombre_bras)

        self.mean_rewards = np.full([p_nombre_bras],0.5)

        self.mean_rewards[random.randint(0,p_nombre_bras-1)] += p_delta

        self.optimal_reward = 0.5 + p_delta

    def _initialization_environment(self):
         pass

    def _perform_action(self, p_action):

        reward = bernoulli.rvs(self.mean_rewards[p_action])
        return reward, self.optimal_reward - reward
