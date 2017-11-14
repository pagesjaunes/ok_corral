import numpy as np
from ok_corral.engine.helper import serialize_json, deserialize_json

class LinearBrain():

    A_CODE = "_A"
    B_CODE = "_b"
    A_INV_CODE = "_A_inv"
    THETA_CODE = "_theta"

    DIMENSION = "dimension"

    def __init__(self, p_dimension = None):

        self.dimension = p_dimension

        self.reset()

    def get_value(self, p_context):
        """

        :param p_context:
        :return: estimation récompense, intervalle de confiance supérieur
        """
        value = np.matmul(np.transpose(self._theta), p_context)
        confidence_interval = np.sqrt(np.matmul(np.matmul(np.transpose(p_context), self._A_inv), p_context))

        return value, value + confidence_interval

    def observe(self, p_context, p_action, p_reward, p_update = True):

        self._A = self._A + np.matmul(p_context, np.transpose(p_context))
        self._b = self._b + p_context * p_reward

        if p_update:
            self._invert()

    def reset(self):

        self.t = 0

        self._A = np.identity(self.dimension)
        self._b = np.zeros((self.dimension, 1))

        self._invert()

    def _invert(self):

        self._A_inv = np.linalg.inv(self._A)
        self._theta = np.matmul(self._A_inv, self._b)

    def to_json(self, p_dump=True):

        dictionary = {self.DIMENSION: self.dimension, self.A_CODE: self._A.tolist(), self.B_CODE: self._b.tolist(),
                             self.A_INV_CODE: self._A_inv.tolist(), self.THETA_CODE: self._theta.tolist()}

        return serialize_json(dictionary, p_dump)

    @staticmethod
    def from_json(p_json):

        dictionary = deserialize_json(p_json)

        linear_brain = LinearBrain(p_dimension= dictionary[LinearBrain.DIMENSION])

        linear_brain._A = np.array(dictionary[LinearBrain.A_CODE])
        linear_brain._b = np.array(dictionary[LinearBrain.B_CODE])
        linear_brain._A_inv = np.array(dictionary[LinearBrain.A_INV_CODE])
        linear_brain._theta = np.array(dictionary[LinearBrain.THETA_CODE])

        linear_brain._invert()

        return linear_brain