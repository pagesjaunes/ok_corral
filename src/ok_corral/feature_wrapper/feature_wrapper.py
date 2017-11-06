import abc
import numpy as np
import json
import itertools

from ok_corral.helper import *

FEATURE_TYPE_REAL = "FT_REAL"
FEATURE_TYPE_CAT_NUMBER = "FT_CAT_NUMBER"
FEATURE_TYPE_CAT_STRING = "FT_CAT_STRING"


AVAILABLE_TYPES = []


class Feature():
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_array(self):

        pass

    @abc.abstractmethod
    def get_array_dimension(self):

        pass

    @abc.abstractmethod
    def get_json(self, p_dump = True):

        pass

    @staticmethod
    @abc.abstractmethod
    def from_json(p_json):

        pass


class RealValuedFeature(Feature):

    def __init__(self, p_dimension, p_name = None):

        self.dimension = p_dimension
        self.type = FEATURE_TYPE_REAL
        self.name = p_name

    def get_array(self, p_json):

        value = deserialize_json(p_json)

        if not type(value) == list:
            value = [value]

        assert len(value) == self.dimension, self.name + " (got: "+str(len(value))+ " want: "+str(self.dimension)+")"

        return list(map(lambda x: float(x), value))

    def get_array_dimension(self):

        return self.dimension

    def to_json(self, p_dump = True):

        dictionary = {"type" : self.type, "dimension" : self.dimension}

        if self.name is not None:
            dictionary["name"] = self.name

        return serialize_json(dictionary, p_dump)

    @staticmethod
    def from_json(p_json):

        dictionary = deserialize_json(p_json)

        return RealValuedFeature(int(dictionary["dimension"]), p_name = dictionary["name"] if "name" in dictionary else None)


# Description
# [ {nom , type, *param} ]

# [ {name: "plop", value: [v1,v2...,vn }]
# [ {name: "plop", value: [v1,v2...,vn }]
# numerique, nombre de dimension
# categoriel, nombre de categoriez

class FeatureWrapper():

    def __init__(self):
        self.features_list = []

        pass

    def add_feature(self,p_feature):

        assert isinstance(p_feature, Feature)

        # TODO Vérifier si la feature existe pas déjà
        # Si oui, lancer une exception
        self.features_list.append(p_feature)


    def add_feature_from_loaded_json(self, p_loaded_json):

        if p_loaded_json["type"] == FEATURE_TYPE_REAL:

            self.add_feature(RealValuedFeature.from_json(p_loaded_json))

        else:
            assert False



    def get_all_features_as_real_valued_array(self, p_context_json):

        # TODO Optimiser ça pour passer des strides sur un seul array

        loaded_json = deserialize_json(p_context_json)

        arrays = []

        for i_json in loaded_json:

            arrays.append(self.get_one_feature_from_loaded_json(i_json))

        return list(itertools.chain.from_iterable(arrays))

    def get_features_as_dictionnary_of_placeholder(self, p_context_json):

        return None

    def get_array_dimension(self):

        return np.sum([i_feature.get_array_dimension for i_feature in self.features_list])

    def to_json(self, p_dump = True):

        return serialize_json(([x.to_json(False) for x in self.features_list]), p_dump)


    @staticmethod
    def from_json(p_json):

        wrapper = FeatureWrapper()

        for i_feature in deserialize_json(p_json):
            wrapper.add_feature_from_loaded_json(i_feature)

        return wrapper