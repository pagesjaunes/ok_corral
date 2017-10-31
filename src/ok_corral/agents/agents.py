import numpy as np
import random
import math
import json


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

    def to_json(self):
        pass

    def from_json(self, p_json):
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

    def to_json(self):

        # Nombre bras
        # Bras i : [succes, echecs]

        json_dictionnary = {}
        json_dictionnary["nombre_bras"] = self.nombre_bras
        json_dictionnary["class"] = type(self).__name__
        json_dictionnary["prior"] = {}

        for k in range(self.nombre_bras):
            json_dictionnary["prior"][k] = {"success" : self.prior[k][self.INDEX_SUCCESS], "failure" : self.prior[k][self.INDEX_FAILURE]}

        return json.dumps(json_dictionnary)


    @staticmethod
    def from_json(p_json):

        json_dictionnary = json.loads(p_json)

        instance = ThompsonSampling(json_dictionnary["nombre_bras"])

        for k in range(instance.nombre_bras):
            instance.prior[k][instance.INDEX_SUCCESS] = int(json_dictionnary["prior"][str(k)]["success"])
            instance.prior[k][instance.INDEX_FAILURE] = int(json_dictionnary["prior"][str(k)]["failure"])

        return instance



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

        # Nombre de tours de jeux
        self.t += 1
        # Nombre de reward
        self.counters[p_action][self.INDEX_NOMBRE_TIRAGES] += 1
        # Récompense cumulée
        self.counters[p_action][self.INDEX_RECOMPENSE_CUMULEE] += p_reward

    def reset(self):
        self.t = 1
        self.counters = np.ones((self.nombre_bras, 2), dtype=np.float64)

    def to_json(self):

        # Nombre bras
        # Compteurs t
        # Bras i : [INDEX_NOMBRE_TIRAGES, INDEX_RECOMPENSE_CUMULEE]

        json_dictionnary = {}
        json_dictionnary["nombre_bras"] = self.nombre_bras

        json_dictionnary["class"] = type(self).__name__

        json_dictionnary["nb_iterations"] = self.t


        json_dictionnary["counters"] = {}

        for k in range(self.nombre_bras):
            json_dictionnary["counters"][k] = {"nb_tirages" : self.counters[k][self.INDEX_NOMBRE_TIRAGES], "recompense_cumulee" : self.counters[k][self.INDEX_RECOMPENSE_CUMULEE]}

        return json.dumps(json_dictionnary)


    @staticmethod
    def from_json(p_json):

        json_dictionnary = json.loads(p_json)

        instance = UCB(json_dictionnary["nombre_bras"])

        instance.t = json_dictionnary["nb_iterations"]

        for k in range(instance.nombre_bras):
            instance.counters[k][instance.INDEX_NOMBRE_TIRAGES] = int(json_dictionnary["counters"][str(k)]["nb_tirages"])
            instance.counters[k][instance.INDEX_RECOMPENSE_CUMULEE] = int(json_dictionnary["counters"][str(k)]["recompense_cumulee"])

        return instance