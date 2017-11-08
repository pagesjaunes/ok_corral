# coding=utf-8
import json
import math
import random

import numpy as np
import abc

from ok_corral.feature_wrapper.feature_wrapper import FeatureWrapper
from ok_corral.helper import *


class Agent:
    __metaclass__ = abc.ABCMeta

    """
    Contrat de l'agent
    """

    NOMBRE_BRAS = "nombre_bras"
    CLASS = "class"

    def __init__(self):
        pass

    @abc.abstractmethod
    def select_action(self, *args):
        pass

    @abc.abstractmethod
    def observe(self, *args):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def to_json(self, p_dump):
        return None

    @staticmethod
    @abc.abstractmethod
    def from_json(self, p_json):
        return None


### Bandits Non Contextuels


class Bandit(Agent):


    def __init__(self):
        Agent.__init__(self)


class RandomBandit(Bandit):
    def __init__(self, p_nombre_bras):
        self.nombre_bras = p_nombre_bras
        Bandit.__init__(self)

    def select_action(self):
        return random.randint(0, self.nombre_bras - 1)

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

    def select_action(self):

        sampling = np.array(
            [np.random.beta(self.prior[i][self.INDEX_SUCCESS], self.prior[i][self.INDEX_FAILURE]) for i in
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

        for k in range(self.nombre_bras):
            json_dictionnary["prior"][k] = {self.SUCCESS: self.prior[k][self.INDEX_SUCCESS],
                                            self.FAILURE: self.prior[k][self.INDEX_FAILURE]}

        return serialize_json(json_dictionnary, p_dump)

    @staticmethod
    def from_json(p_json):

        json_dictionary = deserialize_json(p_json)

        instance = ThompsonSampling(json_dictionary[ThompsonSampling.NOMBRE_BRAS])

        for k in range(instance.nombre_bras):
            instance.prior[k][instance.INDEX_SUCCESS] = int(json_dictionary[ThompsonSampling.PRIOR][str(k)][ThompsonSampling.SUCCESS])
            instance.prior[k][instance.INDEX_FAILURE] = int(json_dictionary[ThompsonSampling.PRIOR][str(k)][ThompsonSampling.FAILURE])

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

    def select_action(self):

        upper_bounds = np.array([self._getUpperConfidenceBound(k) for k in range(self.nombre_bras)])

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


### Bandits Contextuels

class ContextualBandit(Agent):

    WRAPPER = "wrapper"

    def __init__(self):
        Agent.__init__(self)


class RandomContextualBandit(ContextualBandit):
    def __init__(self, p_nombre_bras, p_dimension):
        self.nombre_bras = p_nombre_bras
        self.dimension = p_dimension

        Bandit.__init__(self)

    def select_action(self, p_context):
        return random.randint(0, self.nombre_bras - 1)

    def to_json(self, p_dump=True):
        return serialize_json({self.NOMBRE_BRAS: self.nombre_bras}, p_dump)

    @staticmethod
    def from_json(p_json):
        return RandomContextualBandit(deserialize_json(p_json)[self.NOMBRE_BRAS])


class LinUCB(ContextualBandit):

    A_CODE = "_A"
    B_CODE = "_b"
    A_INV_CODE = "_A_inv"
    THETA_CODE = "_theta"

    DIMENSION = "dimension"

    def __init__(self, p_nombre_bras, p_dimension=None, p_wrapper=None):

        assert p_dimension is not None or p_wrapper is not None

        self.nombre_bras = p_nombre_bras

        self.wrapper = p_wrapper

        if p_wrapper is not None:

            self.dimension = p_wrapper.get_array_dimension()

        else:
            self.dimension = p_dimension

        self.reset()
        Bandit.__init__(self)

        self._tmp_value = np.zeros(self.nombre_bras)

    def select_action(self, p_context):

        p_context = convert_json_to_array_of_reals(p_context, self.wrapper)

        for k in range(self.nombre_bras):
            value = np.matmul(np.transpose(self._theta[k]), p_context)
            confidence_interval = np.sqrt(np.matmul(np.matmul(np.transpose(p_context), self._A_inv[k]), p_context))

            self._tmp_value[k] = value + confidence_interval

        return np.argmax(self._tmp_value)

    def observe(self, p_context, p_action, p_reward):
        p_context = convert_json_to_array_of_reals(p_context, self.wrapper)

        self.t += 1
        self._A[p_action] = self._A[p_action] + np.matmul(p_context, np.transpose(p_context))
        self._b[p_action] = self._b[p_action] + p_context * p_reward

        if self.t % 1000 == 0:
            self._invert()

    def reset(self):

        self.t = 0

        self._A = []
        self._b = []

        for k in range(self.nombre_bras):
            self._A.append(np.identity(self.dimension))
            self._b.append(np.zeros((self.dimension, 1)))

        self._invert()

    def _invert(self):

        self._A_inv = []
        self._theta = []
        for k in range(self.nombre_bras):
            self._A_inv.append(np.linalg.inv(self._A[k]))

            self._theta.append(np.matmul(self._A_inv[k], self._b[k]))

    def to_json(self, p_dump=True):

        dictionary = {self.NOMBRE_BRAS: self.nombre_bras, self.DIMENSION: self.dimension}

        for k in range(self.nombre_bras):
            dictionary[k] = {self.A_CODE: self._A[k].tolist(), self.B_CODE: self._b[k].tolist(),
                             self.A_INV_CODE: self._A_inv[k].tolist(), self.THETA_CODE: self._theta[k].tolist()}

        if self.wrapper is not None:
            dictionary[self.WRAPPER] = self.wrapper.to_json(False)

        return serialize_json(dictionary, p_dump)

    @staticmethod
    def from_json(p_json):

        dictionary = deserialize_json(p_json)
        wrapper = None

        if LinUCB.WRAPPER in dictionary:
            wrapper = FeatureWrapper.from_json(p_json=dictionary[LinUCB.WRAPPER])

        linucb = LinUCB(json.loads(p_json)[LinUCB.NOMBRE_BRAS], p_dimension=dictionary[LinUCB.DIMENSION], p_wrapper=wrapper)

        for k in range(linucb.nombre_bras):
            dic_k = dictionary[str(k)]
            linucb._A[k] = np.array(dic_k[LinUCB.A_CODE])
            linucb._b[k] = np.array(dic_k[LinUCB.B_CODE])
            linucb._A_inv[k] = np.array(dic_k[LinUCB.A_INV_CODE])
            linucb._theta[k] = np.array(dic_k[LinUCB.THETA_CODE])

        return linucb


# Helpers Bandit Contextuel
def convert_json_to_array_of_reals(p_json, p_wrapper):
    if type(p_json) == list:
        p_json = p_wrapper.get_all_features_as_real_valued_array(p_json)

    return p_json
