import json, random

import numpy as np

from ok_corral.engine.agents.agents import Agent
from ok_corral.engine.agents.bandits import Bandit
from ok_corral.engine.feature_wrapper import FeatureWrapper
from ok_corral.engine.helper import serialize_json, deserialize_json


class ContextualBandit(Agent):

    WRAPPER = "wrapper"

    def __init__(self):
        Agent.__init__(self)


class RandomContextualBandit(ContextualBandit):
    def __init__(self, p_nombre_bras, p_dimension):
        self.nombre_bras = p_nombre_bras
        self.dimension = p_dimension

        Bandit.__init__(self)

    def select_action(self, p_context, p_filtre = None):

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
        return RandomContextualBandit(deserialize_json(p_json)[RandomContextualBandit.NOMBRE_BRAS])



class LinUCB(ContextualBandit):

    A_CODE = "_A"
    B_CODE = "_b"
    A_INV_CODE = "_A_inv"
    THETA_CODE = "_theta"

    DIMENSION = "dimension"

    def __init__(self, p_nombre_bras, p_dimension = None, p_wrapper=None, p_specific_context = None):

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

    def select_action(self, p_context, p_filtre = None):

        p_context = if_json_convert_to_array_of_reals(p_context, self.wrapper)

        for i_k in range(self.nombre_bras):

            if p_filtre is None or i_k in p_filtre:
                value = np.matmul(np.transpose(self._theta[i_k]), p_context)
                confidence_interval = np.sqrt(np.matmul(np.matmul(np.transpose(p_context), self._A_inv[i_k]), p_context))

                self._tmp_value[i_k] = value + confidence_interval

            else:

                self._tmp_value[i_k] = -99999

        return np.argmax(self._tmp_value)

    def observe(self, p_context, p_action, p_reward):
        p_context = if_json_convert_to_array_of_reals(p_context, self.wrapper)

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


def if_json_convert_to_array_of_reals(p_json, p_wrapper):
    """
    Si p_json n'est pas un tableau numpy, essaye de le convertir en numpy
    :param p_json: Le contexte, au format numpy, en json ou json loadé
    :param p_wrapper: L'éventuel wrapper pour faire la conversion
    :return:
    """
    if type(p_json) == list:
        p_json = p_wrapper.get_all_features_as_real_valued_array(p_json)

    return p_json