import os
import sys
import time

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

def bandit_vs_env(p_bandit, p_env, p_nb_trial = 20):
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

        reward, regret = p_env.run(p_bandit)

        mean_reward += reward[len(reward)-1][1]
        mean_regret += regret[len(regret)-1][1]

    print("Decisions par secondes :",str(float(reward[len(reward)-1][0])/(float(time.clock() - t_start)/float(p_nb_trial))))

    return float(mean_reward)/float(p_nb_trial), float(mean_regret)/float(p_nb_trial)


def test_stochastic_env():

    env = StochasticBanditEnvironment(20,0.1)

    bandit_random = RandomBandit(20)
    bandit_TS = ThompsonSampling(20)
    bandit_UCB = UCB(20)

    print("Random")
    print(bandit_vs_env(bandit_random, env, 1))
    print("Thompson Sampling")
    print(bandit_vs_env(bandit_TS, env, 1))
    print("UCB")
    print(bandit_vs_env(bandit_UCB, env, 1))

if __name__ == "__main__":

    test_stochastic_env()