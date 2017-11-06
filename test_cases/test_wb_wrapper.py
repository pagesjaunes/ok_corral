import os
import sys
import time
import pickle
import unittest

try:
    dir_path = os.path.abspath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)), os.pardir))
except:
    dir_path = os.getcwd()

if dir_path not in sys.path:
    sys.path.append(dir_path)

from ok_corral.bandits import LinUCB
from ok_corral.feature_wrapper.feature_wrapper import FeatureWrapper, RealValuedFeature


class TestFeature(unittest.TestCase):

    def test_feature_serialization(self):

        feature = RealValuedFeature(p_dimension=5)

        json = feature.to_json()

        feature_from_json = RealValuedFeature.from_json(json).to_json()

        self.assertEqual(json,feature_from_json)


    def test_feature_serialization_with_name(self):

        name = "test_jeaoefahoaeohaeoh"
        feature = RealValuedFeature(p_dimension=5, p_name = name)

        json = feature.to_json()

        self.assertTrue('"name": '+'"'+name+'"' in json)

        feature_from_json = RealValuedFeature.from_json(json).to_json()

        self.assertEqual(json, feature_from_json)

    def test_conversion_to_array_list_real(self):

        name = "test_jeaoefahoaeohaeoh"
        feature = RealValuedFeature(p_dimension=5, p_name = name)

        value = "[2,3,4,5,2]"

        want = [2.,3.,4.,5.,2.]

        for i, i_v in enumerate(feature.get_array(value)):

            self.assertEqual(i_v, want[i])

        try:
            value = "5"

            feature.get_array(value)

            self.assertTrue(False, msg = "On voulait l'exception!")

        except AssertionError:

            pass


    def test_conversion_to_array_one_real(self):

        name = "test_jeaoefahoaeohaeoh"
        feature = RealValuedFeature(p_dimension=1, p_name = name)

        value = "2"

        want = [2.]

        for i, i_v in enumerate(feature.get_array(value)):

            self.assertEqual(i_v, want[i])


class TestWrapper(unittest.TestCase):

    def test_wrapper(self):

        name = "test_jeaoefahoaeohaeoh"
        feature_1 = RealValuedFeature(p_dimension=2, p_name = name)

        name2 = "test2_piefaonefnaonpiaef"
        feature_2 = RealValuedFeature(p_dimension=5, p_name = name2)

        wrapper = FeatureWrapper()
        wrapper.add_feature(feature_1)
        wrapper.add_feature(feature_2)

        json = wrapper.to_json()

        deserialized_wrapper = FeatureWrapper.from_json(json)

        self.assertEqual(json,deserialized_wrapper.to_json())



if __name__ == '__main__':
    unittest.main()

