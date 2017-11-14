import json, random

import numpy as np

from ok_corral.engine.agents.agent import Agent
from ok_corral.engine.agents.bandits import Bandit
from ok_corral.engine.feature_wrapper import FeatureWrapper
from ok_corral.engine.helper import serialize_json, deserialize_json
from ok_corral.engine.agents.brains.linear_brain import LinearBrain



class ContextualBandit(Agent):

    WRAPPER = "wrapper"
    BRAIN = "brain"

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

    def _get_dimension(self, p_k):

        if self.wrappers is not None:

            return self.wrappers.get_array_dimension() if type(self.wrappers) != list else self.wrappers[p_k].get_array_dimension()

        else:

            return self.dimensions if type(self.dimensions) != list else self.dimensions[p_k]

    def _get_wrapper(self, p_k):

        if self.wrappers is not None:

            return self.wrappers if type(self.wrappers) != list else self.wrappers[p_k]

        else:
            return None

    def __init__(self, p_nombre_bras, p_dimensions = None, p_wrappers=None):

        assert p_dimensions is not None or p_wrappers is not None

        self.nombre_bras = p_nombre_bras
        self.dimensions = p_dimensions
        self.wrappers = p_wrappers
        self.brains = []
        self.counters = []

        for i_k in range(p_nombre_bras):

            self.counters.append(0)

            dimension = self._get_dimension(i_k)

            brain = LinearBrain(dimension)

            self.brains.append(brain)

        Bandit.__init__(self)

        self._tmp_value = np.zeros(self.nombre_bras)

    def select_action(self, p_context, p_filtre = None):

        # Un seul contexte
        if type(p_context[0]) != list:

            for i_k in range(self.nombre_bras):

                p_context = if_json_convert_to_array_of_reals(p_context, self._get_wrapper(i_k))

                if p_filtre is None or i_k in p_filtre:

                    self._tmp_value[i_k] = self.brains[i_k].get_value(p_context)[1]

                else:

                    self._tmp_value[i_k] = -99999

            return np.argmax(self._tmp_value)

        # Contextes multiples

        else:
            ucb = []
            for i_k, i_context in p_context:

                if p_filtre is None or i_k in p_filtre:

                    context = if_json_convert_to_array_of_reals(i_context[i_context], self._get_wrapper(i_k))

                    ucb.append(self.brains[i_k](context))

                else:

                    ucb.append(-99999)

            return np.argmax(ucb)



    def observe(self, p_context, p_action, p_reward):

        self.counters[p_action] += 1
        p_context = if_json_convert_to_array_of_reals(p_context, self._get_wrapper(p_action))

        self.brains[p_action].observe(p_context,p_action,p_reward, self.counters[p_action]%1000 == 0)


    def to_json(self, p_dump=True):

        dictionary = {self.NOMBRE_BRAS: self.nombre_bras}

        is_wrapper_unique = self.wrappers is not None and type(self.wrappers) != list

        if is_wrapper_unique:
            dictionary[self.WRAPPER] = self.wrappers.to_json(False)

        for i_k in range(self.nombre_bras):

            dictionary[i_k] = {}

            dictionary[i_k][self.BRAIN] = self.brains[i_k].to_json(False)

            if self.wrappers is not None:

                if not is_wrapper_unique:
                    dictionary[i_k][self.WRAPPER] = self.wrappers[i_k].to_json(False)

        return serialize_json(dictionary, p_dump)

    @staticmethod
    def from_json(p_json):

        dictionary = deserialize_json(p_json)
        wrappers = None

        if LinUCB.WRAPPER in dictionary:
            wrappers = FeatureWrapper.from_json(p_json=dictionary[LinUCB.WRAPPER])

        nombre_bras = int(dictionary[LinUCB.NOMBRE_BRAS])

        brains = []
        for i_k in range(nombre_bras):

            dic_k = dictionary[str(i_k)]

            brains.append(LinearBrain.from_json(dic_k[LinUCB.BRAIN]))

            if LinUCB.WRAPPER in dic_k:
                if wrappers is None:
                    wrappers = [FeatureWrapper.from_json(p_json=dic_k[LinUCB.WRAPPER])]
                else :
                    wrappers.append(FeatureWrapper.from_json(p_json=dic_k[LinUCB.WRAPPER]))

        linucb = LinUCB(json.loads(p_json)[LinUCB.NOMBRE_BRAS], p_dimensions=1, p_wrappers=wrappers)
        linucb.brains = brains

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