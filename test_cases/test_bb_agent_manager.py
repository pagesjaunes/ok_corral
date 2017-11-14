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

from ok_corral.engine.agent_manager import AgentManager, PrivilegeException
from ok_corral.engine.feature_wrapper import FeatureWrapper, RealValuedFeature

class TestAgentManager(unittest.TestCase):

    def test_creation_rights(self):

        manager = AgentManager()

        try:
            manager.add_bandit("ma clé", "mon instance", "ts", 5)

            self.assertTrue(False)

        except PrivilegeException:
            pass


    def test_end_to_end_bandit(self):

        manager = AgentManager()
        manager.privilege_manager._user_keys.add("ma clé")

        key = manager.add_bandit("ma clé", "mon instance", "ts", 5)

        action = manager.get_decision(key)

        manager.observe(key,action,1)

        self.assertEqual(manager.instances[key]["instance"].prior[action][0], 2)


    def test_end_to_end_bandit_contextuel(self):

        manager = AgentManager()
        manager.privilege_manager._user_keys.add("ma clé")

        name = "test_jeaoefahoaeohaeoh"
        feature_1 = RealValuedFeature(p_dimension=2, p_name=name)
        name2 = "test2_piefaonefnaonpiaef"
        feature_2 = RealValuedFeature(p_dimension=5, p_name=name2)

        wrapper = FeatureWrapper()
        wrapper.add_feature(feature_1)
        wrapper.add_feature(feature_2)

        json_contexte = wrapper.to_json()

        key = manager.add_bandit("ma clé", "mon instance", "linucb", 5, json_contexte)


        contexte = [{"value" : "[2,3]"},{"value" : "[2,3,4,5,2]"}]

        action = manager.get_decision(key,contexte)

        manager.observe(key,action,1,contexte)

        #TODO Checker le comportement de LinUCB plus en profondeur

        self.assertTrue(True, "Executer, c'est tester.")

if __name__ == '__main__':
    unittest.main()

