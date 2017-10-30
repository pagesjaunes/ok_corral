import numpy as np
import random
import math


class Agent:
    """
    Contrat de l'agent
    """

    def __init__(self):
        pass

    def select_action(self):
        pass

    def observe(self, *args):
        pass

    def reset(self):
        pass


### Bandits Non Contextuels


class Bandit(Agent):

    def __init__(self):

        Agent.__init__(self)



class RandomBandit(Bandit):

    def __init__(self, p_nombre_bras):

        self.nombre_bras = p_nombre_bras
        Bandit.__init__(self)

    def select_action(self):

        return random.randint(0,self.nombre_bras - 1)


class ThompsonSampling(Bandit):

    def __init__(self, p_nombre_bras):

        Bandit.__init__(self)

        self.INDEX_SUCCESS = 0
        self.INDEX_FAILURE = 1

        self.nombre_bras = p_nombre_bras
        self.prior = None
        self.reset()

    def select_action(self):

        sampling = np.array([np.random.beta(self.prior[i][self.INDEX_SUCCESS], self.prior[i][self.INDEX_FAILURE]) for i in range(self.nombre_bras)])

        return np.argmax(sampling)

    def observe(self, p_action, p_reward):

        super(ThompsonSampling, self).observe(p_action, p_reward)
        # Nombre de succès
        self.prior[p_action][self.INDEX_SUCCESS] += max(p_reward, 0)
        # Nombre d'échecs
        self.prior[p_action][self.INDEX_FAILURE] += max(1 - p_reward, 0)

    def reset(self):
        self.prior = np.ones((self.nombre_bras, 2), dtype=np.float64)


class UCB(Bandit):

    """
    Auer 2002, Finite-time Analysis of the Multiarmed Bandit Problem
    """

    def __init__(self, p_nombre_bras):

        Bandit.__init__(self)

        self.INDEX_NOMBRE_TIRAGES = 0
        self.INDEX_RECOMPENSE_CUMULEE = 1

        self.t = 1
        self.nombre_bras = p_nombre_bras
        self.counters = None
        self.reset()

    def _getUpperConfidenceBound(self, p_index_bras):

        moyenne = self.counters[p_index_bras][self.INDEX_RECOMPENSE_CUMULEE]/self.counters[p_index_bras][self.INDEX_NOMBRE_TIRAGES]

        ucb = math.sqrt(2 * math.log(self.t) / self.counters[p_index_bras][self.INDEX_NOMBRE_TIRAGES])

        return  moyenne + ucb


    def select_action(self):

        upper_bounds = np.array([self._getUpperConfidenceBound(k) for k in range(self.nombre_bras)])

        return np.argmax(upper_bounds)

    def observe(self, p_action, p_reward):

        super(UCB, self).observe(p_action, p_reward)
        # Nombre de reward
        self.counters[p_action][self.INDEX_NOMBRE_TIRAGES] += 1
        # Récompense cumulée
        self.counters[p_action][self.INDEX_RECOMPENSE_CUMULEE] += p_reward

    def reset(self):
        self.t = 1
        self.counters = np.ones((self.nombre_bras, 2), dtype=np.float64)
