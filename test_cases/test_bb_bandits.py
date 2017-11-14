import os
import sys
import time
import pickle
import unittest

try:
    dir_path = os.path.abspath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)), os.pardir))
except:
    dir_path = os.getcwd()

if dir_path not in sys.path:
    sys.path.append(dir_path)


print(dir_path)
from environments.bandit_environment import StochasticBanditEnvironment
from environments.contextual_bandit_environment import Adult

from ok_corral.bandits import ThompsonSampling
from ok_corral.bandits import UCB
from ok_corral.bandits import RandomBandit
from ok_corral.bandits import LinUCB
from ok_corral.bandits import RandomContextualBandit


class TestBanditAlgorithms(unittest.TestCase):

    def test_ucb(self):

        nombre_bras = 20
        env = StochasticBanditEnvironment(nombre_bras,0.2)

        bandit_UCB = UCB(nombre_bras)
        dec_per_sec, reward, regret = bandit_vs_env(bandit_UCB, env, p_nb_trial=1)

        self.assertGreater(dec_per_sec, 4000)
        self.assertLess(regret, 2500)

        print("Stochastic20:UCB",dec_per_sec, reward, regret)

        nombre_bras = 2
        env = StochasticBanditEnvironment(nombre_bras,0.1)

        bandit_UCB = UCB(nombre_bras)
        dec_per_sec, reward, regret = bandit_vs_env(bandit_UCB, env, p_nb_trial=1)

        self.assertGreater(dec_per_sec, 7000)
        self.assertLess(regret, 600)


    def test_thompson_sampling(self):

        nombre_bras = 20

        env = StochasticBanditEnvironment(nombre_bras,0.2)

        bandit_ts = ThompsonSampling(nombre_bras)
        dec_per_sec, reward, regret =  bandit_vs_env(bandit_ts, env, p_nb_trial=1)

        self.assertGreater(dec_per_sec, 4000)
        self.assertLess(regret, 500)
        print("Stochastic20:TS",dec_per_sec, reward, regret)


        nombre_bras = 2
        env = StochasticBanditEnvironment(nombre_bras,0.1)

        bandit_TS = ThompsonSampling(nombre_bras)
        dec_per_sec, reward, regret = bandit_vs_env(bandit_TS, env, p_nb_trial=1)

        self.assertGreater(dec_per_sec, 7000)
        self.assertLess(regret, 300)
        print("Stochastic2:TS",dec_per_sec, reward, regret)



    def test_linUCB(self):

        env = Adult()

        bandit_linUCB = LinUCB(env.get_actions_description(),env.get_environment_description())
        dec_per_sec, reward, regret = bandit_vs_env(bandit_linUCB, env, p_horizon = 50000, p_nb_trial=1)

        print("Adult:LinUCB",dec_per_sec, reward, regret)

        self.assertGreater(dec_per_sec, 2000)
        self.assertLess(regret, 75000)

    def test_RandomContextual(self):

        env = Adult()

        bandit_rand = RandomContextualBandit(env.get_actions_description(),env.get_environment_description())
        dec_per_sec, reward, regret = bandit_vs_env(bandit_rand, env,p_horizon = 50000, p_nb_trial=1)

        print("Adult:Random",dec_per_sec, reward, regret)


def bandit_vs_env(p_bandit, p_env, p_horizon = 100000, p_nb_trial=20):
    """
    @param p_bandit:
    :param p_horizon: Le nombre d'iterations pour une expérience
    :type p_env: Adult
    :param p_env:
    :param p_nb_trial: Le nombre de fois où l'expérience est répétée
    :return:
    """

    mean_reward = 0
    mean_regret = 0

    t_start = time.clock()
    for i in range(p_nb_trial):
        p_bandit.reset()

        reward, regret = p_env.run(p_bandit,p_horizon)

        mean_reward += reward[len(reward) - 1][1]
        mean_regret += regret[len(regret) - 1][1]


    return float(reward[len(reward) - 1][0]) / (float(time.clock() - t_start) / float(p_nb_trial)), float(mean_reward) / float(p_nb_trial), float(mean_regret) / float(p_nb_trial)


if __name__ == '__main__':
    unittest.main()

