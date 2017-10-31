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

from environments.environments import StochasticBanditEnvironment
from ok_corral.bandits import ThompsonSampling
from ok_corral.bandits import UCB
from ok_corral.bandits import RandomBandit


class TestBanditAlgorithms(unittest.TestCase):

    def bandit_vs_env(self, p_bandit, p_env, p_horizon = 100000, p_nb_trial=20):
        """
        @param p_bandit:
        :type p_env: StochasticBanditEnvironment
        :param p_env:
        :param p_nb_trial:
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

    def test_ucb(self):

        env = StochasticBanditEnvironment(20,0.2)

        bandit_UCB = UCB(20)
        dec_per_sec, reward, regret = self.bandit_vs_env(bandit_UCB, env, p_nb_trial=1)

        self.assertGreater(dec_per_sec, 4000)
        self.assertLess(regret, 2500)

        env = StochasticBanditEnvironment(2,0.1)

        bandit_UCB = UCB(2)
        dec_per_sec, reward, regret = self.bandit_vs_env(bandit_UCB, env, p_nb_trial=1)

        self.assertGreater(dec_per_sec, 7000)
        self.assertLess(regret, 600)

    def test_ucb_json(self):

        nombre_bras = 20

        env = StochasticBanditEnvironment(nombre_bras, 0.2)

        bandit_ucb = UCB(nombre_bras)

        _ = self.bandit_vs_env(bandit_ucb, env, 100, 1)

        jsonified_ucb = bandit_ucb.to_json()

        from_json = UCB.from_json(jsonified_ucb)

        self.assertEqual(bandit_ucb.nombre_bras, from_json.nombre_bras)
        self.assertEqual(bandit_ucb.t, from_json.t)

        for k in range(bandit_ucb.nombre_bras):
            self.assertEqual(bandit_ucb.counters[k][0], from_json.counters[k][0])
            self.assertEqual(bandit_ucb.counters[k][1], from_json.counters[k][1])

        self.assertEqual(nombre_bras, k + 1)


    def test_thompson_sampling(self):

        nombre_bras = 20

        env = StochasticBanditEnvironment(nombre_bras,0.2)

        bandit_ts = ThompsonSampling(nombre_bras)
        dec_per_sec, reward, regret =  self.bandit_vs_env(bandit_ts, env, p_nb_trial=1)

        self.assertGreater(dec_per_sec, 4000)
        self.assertLess(regret, 500)

        env = StochasticBanditEnvironment(2,0.1)

        bandit_TS = ThompsonSampling(2)
        dec_per_sec, reward, regret = self.bandit_vs_env(bandit_TS, env, p_nb_trial=1)

        self.assertGreater(dec_per_sec, 7000)
        self.assertLess(regret, 250)

    def test_thompson_sampling_json(self):
        nombre_bras = 20

        env = StochasticBanditEnvironment(nombre_bras,0.2)

        bandit_ts = ThompsonSampling(nombre_bras)
        _ =  self.bandit_vs_env(bandit_ts, env, 100, 1)

        jsonified_ts = bandit_ts.to_json()

        from_json = ThompsonSampling.from_json(jsonified_ts)

        self.assertEqual(bandit_ts.nombre_bras, from_json.nombre_bras)

        for k in range(bandit_ts.nombre_bras):

            self.assertEqual(bandit_ts.prior[k][0], from_json.prior[k][0])
            self.assertEqual(bandit_ts.prior[k][1], from_json.prior[k][1])

        self.assertEqual(nombre_bras,k+1)

if __name__ == '__main__':
    unittest.main()

