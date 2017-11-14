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
from test_bb_bandits import bandit_vs_env

from ok_corral.bandits import ThompsonSampling
from ok_corral.bandits import UCB
from ok_corral.bandits import RandomBandit
from ok_corral.bandits import LinUCB
from ok_corral.bandits import RandomContextualBandit


class TestBanditAlgorithms(unittest.TestCase):


    def test_filtre_ucb(self):

        nombre_bras = 20
        bandit_UCB = UCB(nombre_bras)
        filtre = set([5, 6, 9, 12])

        for _ in range(100000):
            self.assertTrue(bandit_UCB.select_action(p_filtre=filtre) in filtre)

    def test_ucb_json(self):

        nombre_bras = 20

        env = StochasticBanditEnvironment(nombre_bras, 0.2)

        bandit_ucb = UCB(nombre_bras)

        _ = bandit_vs_env(bandit_ucb, env, 100, 1)

        jsonified_ucb = bandit_ucb.to_json()

        from_json = UCB.from_json(jsonified_ucb)

        self.assertEqual(bandit_ucb.nombre_bras, from_json.nombre_bras)
        self.assertEqual(bandit_ucb.t, from_json.t)

        for k in range(bandit_ucb.nombre_bras):
            self.assertEqual(bandit_ucb.counters[k][0], from_json.counters[k][0])
            self.assertEqual(bandit_ucb.counters[k][1], from_json.counters[k][1])

        self.assertEqual(nombre_bras, k + 1)


    def test_filtre_ts(self):

        nombre_bras = 20
        bandit_ts = ThompsonSampling(nombre_bras)
        filtre = set([5, 6, 9, 12, 13, 19])

        for _ in range(100000):
            self.assertTrue(bandit_ts.select_action(p_filtre=filtre) in filtre)


    def test_thompson_sampling_json(self):

        nombre_bras = 20

        env = StochasticBanditEnvironment(nombre_bras,0.2)

        bandit_ts = ThompsonSampling(nombre_bras)
        _ = bandit_vs_env(bandit_ts, env, 100, 1)

        jsonified_ts = bandit_ts.to_json()

        from_json = ThompsonSampling.from_json(jsonified_ts)

        self.assertEqual(bandit_ts.nombre_bras, from_json.nombre_bras)

        for k in range(bandit_ts.nombre_bras):

            self.assertEqual(bandit_ts.prior[k][0], from_json.prior[k][0])
            self.assertEqual(bandit_ts.prior[k][1], from_json.prior[k][1])

        self.assertEqual(nombre_bras,k+1)

if __name__ == '__main__':
    unittest.main()

