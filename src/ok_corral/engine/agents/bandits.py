import random, math

import numpy as np

from ok_corral.engine.agents.agent import Agent
from ok_corral.engine.helper import serialize_json, deserialize_json


class Bandit(Agent):

    def __init__(self):
        Agent.__init__(self)

class RandomBandit(Bandit):
    def __init__(self, p_nombre_bras):
        self.nombre_bras = p_nombre_bras
        Bandit.__init__(self)

    def select_action(self, p_filtre = None):

        if p_filtre is None:

            return random.randint(0, self.nombre_bras - 1)

        else:

            if type(p_filtre) != list:
                p_filtre = list(p_filtre)

            return p_filtre[random.randint(0, len(p_filtre) - 1)]

    def to_json(self, p_dump=True):

        return serialize_json({self.NOMBRE_BRAS: self.nombre_bras}, p_dump)

    @staticmethod
    def from_json(p_json):
        dic = deserialize_json(p_json)

        return RandomBandit(dic[RandomBandit.NOMBRE_BRAS])


class ThompsonSampling(Bandit):

    INDEX_SUCCESS = 0
    INDEX_FAILURE = 1
    SUCCESS = "success"
    FAILURE = "failure"
    PRIOR = "prior"

    def __init__(self, p_nombre_bras):

        Bandit.__init__(self)

        self.nombre_bras = p_nombre_bras
        self.prior = None
        self.reset()

    def select_action(self, p_filtre = None):

        sampling = np.array(
            [np.random.beta(self.prior[i_k][self.INDEX_SUCCESS], self.prior[i_k][self.INDEX_FAILURE]) if (p_filtre is None or i_k in p_filtre) else 0 for i_k in
             range(self.nombre_bras)])

        return np.argmax(sampling)

    def observe(self, p_action, p_reward):

        super(ThompsonSampling, self).observe(p_action, p_reward)
        # Nombre de succès
        self.prior[p_action][self.INDEX_SUCCESS] += max(p_reward, 0)
        # Nombre d'échecs
        self.prior[p_action][self.INDEX_FAILURE] += max(1 - p_reward, 0)

    def reset(self):
        self.prior = np.ones((self.nombre_bras, 2), dtype=np.float64)

    def to_json(self, p_dump=True):

        # Nombre bras
        # Bras i : [succes, echecs]

        json_dictionnary = {}
        json_dictionnary[self.NOMBRE_BRAS] = self.nombre_bras
        json_dictionnary[self.CLASS] = type(self).__name__
        json_dictionnary["prior"] = {}

        for i_k in range(self.nombre_bras):
            json_dictionnary["prior"][i_k] = {self.SUCCESS: self.prior[i_k][self.INDEX_SUCCESS],
                                            self.FAILURE: self.prior[i_k][self.INDEX_FAILURE]}

        return serialize_json(json_dictionnary, p_dump)

    @staticmethod
    def from_json(p_json):

        json_dictionary = deserialize_json(p_json)

        instance = ThompsonSampling(json_dictionary[ThompsonSampling.NOMBRE_BRAS])

        for i_k in range(instance.nombre_bras):
            instance.prior[i_k][instance.INDEX_SUCCESS] = int(json_dictionary[ThompsonSampling.PRIOR][str(i_k)][ThompsonSampling.SUCCESS])
            instance.prior[i_k][instance.INDEX_FAILURE] = int(json_dictionary[ThompsonSampling.PRIOR][str(i_k)][ThompsonSampling.FAILURE])

        return instance


class UCB(Bandit):
    """
    Auer 2002, Finite-time Analysis of the Multiarmed Bandit Problem
    """
    INDEX_NOMBRE_TIRAGES = 0
    INDEX_RECOMPENSE_CUMULEE = 1
    NB_TIRAGES = "nb_tirages"
    NB_ITERATIONS_GLOBAL = "nb_iterations"
    COUNTERS = "counters"
    CUMULATED_REWARD = "recompense_cumulee"

    def __init__(self, p_nombre_bras):

        Bandit.__init__(self)

        self.t = 1
        self.nombre_bras = p_nombre_bras
        self.counters = None
        self.reset()

    def _getUpperConfidenceBound(self, p_index_bras):

        moyenne = self.counters[p_index_bras][self.INDEX_RECOMPENSE_CUMULEE] / self.counters[p_index_bras][
            self.INDEX_NOMBRE_TIRAGES]

        ucb = math.sqrt(2 * math.log(self.t) / self.counters[p_index_bras][self.INDEX_NOMBRE_TIRAGES])

        return moyenne + ucb

    def select_action(self, p_filtre = None):

        upper_bounds = np.array([self._getUpperConfidenceBound(i_k) if (p_filtre is None or i_k in p_filtre) else 0 for i_k in range(self.nombre_bras)])

        return np.argmax(upper_bounds)

    def observe(self, p_action, p_reward):

        super(UCB, self).observe(p_action, p_reward)

        # Nombre de tours de jeux
        self.t += 1
        # Nombre de reward
        self.counters[p_action][self.INDEX_NOMBRE_TIRAGES] += 1
        # Récompense cumulée
        self.counters[p_action][self.INDEX_RECOMPENSE_CUMULEE] += p_reward

    def reset(self):
        self.t = 1
        self.counters = np.ones((self.nombre_bras, 2), dtype=np.float64)

    def to_json(self, p_dump=True):

        # Nombre bras
        # Compteurs t
        # Bras i : [INDEX_NOMBRE_TIRAGES, INDEX_RECOMPENSE_CUMULEE]

        json_dictionnary = {}
        json_dictionnary[self.NOMBRE_BRAS] = self.nombre_bras

        json_dictionnary[self.CLASS] = type(self).__name__

        json_dictionnary[self.NB_ITERATIONS_GLOBAL] = self.t

        json_dictionnary[self.COUNTERS] = {}

        for k in range(self.nombre_bras):
            json_dictionnary[self.COUNTERS][k] = {self.NB_TIRAGES: self.counters[k][self.INDEX_NOMBRE_TIRAGES],
                                               self.CUMULATED_REWARD: self.counters[k][self.INDEX_RECOMPENSE_CUMULEE]}

        return serialize_json(json_dictionnary, p_dump)

    @staticmethod
    def from_json(p_json):

        json_dictionary = deserialize_json(p_json)

        instance = UCB(json_dictionary[UCB.NOMBRE_BRAS])

        instance.t = json_dictionary[UCB.NB_ITERATIONS_GLOBAL]

        for k in range(instance.nombre_bras):
            instance.counters[k][instance.INDEX_NOMBRE_TIRAGES] = int(json_dictionary[UCB.COUNTERS][str(k)][UCB.NB_TIRAGES])
            instance.counters[k][instance.INDEX_RECOMPENSE_CUMULEE] = int(
                json_dictionary[UCB.COUNTERS][str(k)][UCB.CUMULATED_REWARD])

        return instance