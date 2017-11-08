import os
import sys
import unittest

try:
    dir_path = os.path.abspath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)), os.pardir))
except:
    dir_path = os.getcwd()

if dir_path not in sys.path:
    sys.path.append(dir_path)

from ok_corral.engine.feature_wrapper import FeatureWrapper, RealValuedFeature, CategoricallyNumberedFeature


class TestFeature(unittest.TestCase):

    def _serialization(self, p_feature):

        json = p_feature.to_json()
        feature_from_json = p_feature.from_json(json).to_json()
        self.assertEqual(json,feature_from_json)

    def test_feature_serialization(self):

        self._serialization(RealValuedFeature(p_dimension=5))
        self._serialization(CategoricallyNumberedFeature(p_cardinality=5))

    def _serialization_with_name(self, p_feature, p_name):
        json = p_feature.to_json()
        self.assertTrue('"name": ' + '"' + p_name + '"' in json)
        feature_from_json = p_feature.from_json(json).to_json()
        self.assertEqual(json, feature_from_json)

    def test_feature_serialization_with_name(self):

        name = "test_jeaoefahoaeohaeoh"
        self._serialization_with_name(RealValuedFeature(p_dimension=5, p_name = name),name)
        self._serialization_with_name(CategoricallyNumberedFeature(p_cardinality=5, p_name = name),name)



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

    def test_conversion_to_array_from_number_categorical(self):

        name = "test_jeaoefahoaeohaeoh"
        feature = CategoricallyNumberedFeature(p_cardinality=5, p_name = name)

        value = "2"

        want = [0, 0, 1, 0, 0]

        for i, i_v in enumerate(feature.get_array(value)):

            self.assertEqual(i_v, want[i])

        value = "3"

        want = [0, 0,0 ,1 ,0]

        for i, i_v in enumerate(feature.get_array(value)):

            self.assertEqual(i_v, want[i])


class TestWrapper(unittest.TestCase):

    def test_wrapper(self):

        name = "my name 1"
        feature_1 = RealValuedFeature(p_dimension=2, p_name = name)

        feature_2 = RealValuedFeature(p_dimension=5)

        name2 = "my name 2"
        feature_3 = CategoricallyNumberedFeature(p_cardinality=5, p_name = name2)

        wrapper = FeatureWrapper()
        wrapper.add_feature(feature_1)
        wrapper.add_feature(feature_2)
        wrapper.add_feature(feature_3)

        json = wrapper.to_json()

        deserialized_wrapper = FeatureWrapper.from_json(json)

        self.assertEqual(json,deserialized_wrapper.to_json())

        self.assertTrue(deserialized_wrapper.features_list[0].dimension == 2)
        self.assertTrue(deserialized_wrapper.features_list[1].dimension == 5)



if __name__ == '__main__':
    unittest.main()

