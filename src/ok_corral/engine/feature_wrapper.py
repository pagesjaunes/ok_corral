import abc
import itertools

import numpy as np

from ok_corral.engine.helper import *


class Feature():
    __metaclass__ = abc.ABCMeta

    FEATURE_TYPE_REAL = "FT_REAL"
    FEATURE_TYPE_CAT_NUMBER = "FT_CAT_NUMBER"
    #FEATURE_TYPE_CAT_STRING = "FT_CAT_STRING"

    AVAILABLE_TYPES = [FEATURE_TYPE_REAL,FEATURE_TYPE_CAT_NUMBER]

    TYPE = "type"
    NAME = "name"


    @abc.abstractmethod
    def get_array(self):
        pass

    @abc.abstractmethod
    def get_array_dimension(self):
        pass

    @abc.abstractmethod
    def to_json(self, p_dump=True):
        pass

    @staticmethod
    @abc.abstractmethod
    def from_json(p_json):
        pass


class CategoricallyNumberedFeature(Feature):

    CARDINALITY = "cardinality"

    def __init__(self, p_cardinality, p_name=None):
        self.cardinality = p_cardinality
        self.type = self.FEATURE_TYPE_CAT_NUMBER
        self.name = p_name

    def get_array(self, p_json):
        value = deserialize_json(p_json)

        assert self.cardinality > value, self.name + " (got: " + str(value) + " want < " + str(self.cardinality) + ")"

        return np.array([(1 if i == value else 0) for i in range(self.cardinality)])

    def get_array_dimension(self):
        return self.cardinality

    def to_json(self, p_dump=True):
        dictionary = {self.TYPE: self.type, self.CARDINALITY: self.cardinality}

        if self.name is not None:
            dictionary[self.NAME] = self.name

        return serialize_json(dictionary, p_dump)

    @staticmethod
    def from_json(p_json):
        dictionary = deserialize_json(p_json)

        return CategoricallyNumberedFeature(int(dictionary[CategoricallyNumberedFeature.CARDINALITY]),
                                            p_name=dictionary[CategoricallyNumberedFeature.NAME] if CategoricallyNumberedFeature.NAME in dictionary else None)


class RealValuedFeature(Feature):

    DIMENSION = "dimension"

    def __init__(self, p_dimension, p_name=None):

        self.dimension = p_dimension
        self.type = self.FEATURE_TYPE_REAL
        self.name = p_name

    def get_array(self, p_json):

        value = deserialize_json(p_json)

        if not type(value) == list:
            value = [value]

        assert len(value) == self.dimension, self.name + " (got: " + str(len(value)) + " want: " + str(
            self.dimension) + ")"

        return list(map(lambda x: float(x), value))

    def get_array_dimension(self):

        return self.dimension

    def to_json(self, p_dump=True):

        dictionary = {self.TYPE: self.type, self.DIMENSION: self.dimension}

        if self.name is not None:
            dictionary[self.NAME] = self.name

        return serialize_json(dictionary, p_dump)

    @staticmethod
    def from_json(p_json):

        dictionary = deserialize_json(p_json)
        return RealValuedFeature(int(dictionary[RealValuedFeature.DIMENSION]),
                                 p_name=dictionary[RealValuedFeature.NAME] if RealValuedFeature.NAME in dictionary else None)


# Description
# [ {nom , type, *param} ]

# [ {name: "plop", value: [v1,v2...,vn }]
# [ {name: "plop", value: [v1,v2...,vn }]
# numerique, nombre de dimension
# categoriel, nombre de categoriez

class FeatureWrapper():

    VALUE = "value"

    def __init__(self):
        self.features_list = []

    def add_feature(self, p_feature):

        assert isinstance(p_feature, Feature)

        # TODO Vérifier si la feature existe pas déjà
        # Si oui, lancer une exception
        self.features_list.append(p_feature)

    def add_feature_from_loaded_json(self, p_loaded_json):

        if p_loaded_json[Feature.TYPE] == Feature.FEATURE_TYPE_REAL:

            self.add_feature(RealValuedFeature.from_json(p_loaded_json))

        elif p_loaded_json[Feature.TYPE] == Feature.FEATURE_TYPE_CAT_NUMBER:
            self.add_feature(CategoricallyNumberedFeature.from_json(p_loaded_json))

        else:
            assert False

    def get_all_features_as_real_valued_array(self, p_context_json):

        # TODO Optimiser ça pour passer des strides sur un seul array

        loaded_json = deserialize_json(p_context_json)

        arrays = []

        for i, i_json in enumerate(loaded_json):
            arrays.append(self.features_list[i].get_array(i_json[self.VALUE] if type(i_json) is dict else i_json))

        concat = list(itertools.chain.from_iterable(arrays))

        return np.reshape(np.array(concat), (len(concat), 1))

    def get_features_as_dictionnary_of_placeholder(self, p_context_json):

        return None

    def get_array_dimension(self):

        return np.sum([i_feature.get_array_dimension() for i_feature in self.features_list])

    def to_json(self, p_dump=True):

        return serialize_json(([x.to_json(False) for x in self.features_list]), p_dump)

    @staticmethod
    def from_json(p_json):

        wrapper = FeatureWrapper()

        for i_feature in deserialize_json(p_json):
            wrapper.add_feature_from_loaded_json(i_feature)

        return wrapper
