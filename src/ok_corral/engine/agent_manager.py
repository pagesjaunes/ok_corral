import os
from binascii import hexlify

from ok_corral.bandits import *
from ok_corral.engine import persistance_manager
from ok_corral.engine.feature_wrapper import FeatureWrapper

# Type d'algorithmes
TYPE_BANDIT = "bandit"
TYPE_BANDIT_CONTEXTUEL = "bandit_contextuel"

SQUELETON_KEY = "skeleton_key_42"


def _key_generation(p_agent):
    key = hexlify(os.urandom(256)).decode()

    while key in p_agent.instances:
        key = hexlify(os.urandom(256)).decode()

    return key


class PrivilegeException(Exception):
    pass


class PrivilegeManager:
    def __init__(self):

        self._user_owned = {}
        if not persistance_manager.check_database():
            persistance_manager.add_user_key_to_database(SQUELETON_KEY, SQUELETON_KEY)

        self._user_keys = persistance_manager.get_user_keys_from_database()

    def add_user(self, p_key, p_name):
        persistance_manager.add_user_key_to_database(p_key, p_name)
        self._user_keys.add(p_key)

    def can_create(self, p_user_key):

        if p_user_key in self._user_keys:

            return True
        else:
            return False

    def register_instance(self, p_user_key, p_instance_key):

        if p_user_key not in self._user_owned:
            self._user_owned[p_user_key] = []

        self._user_owned[p_user_key].append(p_instance_key)


class AgentManager:

    NAME = "name"
    INSTANCE = "instance"
    TYPE = "type"
    ALGO_NAME = "algorithme"
    OWNER_KEY = "owner_key"
    INSTANCE_KEY = "instance_key"

    def __init__(self):

        self.privilege_manager = PrivilegeManager()
        self.instances = persistance_manager.get_instances_from_database()

        for key, value in self.instances.items():
            self.privilege_manager.register_instance(key, self.instances[key])

    # Gestion d'instance
    def add_bandit(self, p_user_key, p_instance_name, p_algorithm_name, p_nombre_bras, p_context_description=None):

        if not self.privilege_manager.can_create(p_user_key):
            raise PrivilegeException("La clé utilisateur n'a pas les droits de création")

        key_generated = _key_generation(self)

        instance = None
        type = None

        if p_context_description is None:

            type = TYPE_BANDIT

            if p_algorithm_name == UPPER_CONFIDENCE_BOUND:
                instance = UCB(p_nombre_bras)

            elif p_algorithm_name == RANDOM_BANDIT:

                instance = RandomBandit(p_nombre_bras)

            elif p_algorithm_name == THOMPSON_SAMPING:

                instance = ThompsonSampling(p_nombre_bras)

            else:

                assert "Algorithme de bandits non trouvé. Ne pas oublier d'inclure la description du contexte si le bandit est contextuel."

        else:

            type = TYPE_BANDIT_CONTEXTUEL
            wrapper = FeatureWrapper.from_json(p_context_description)

            if p_algorithm_name == LINUCB:

                instance = LinUCB(p_nombre_bras, None, wrapper)

            else:

                assert False

        self.instances[key_generated] = {self.INSTANCE: instance, self.TYPE: type, self.ALGO_NAME: p_algorithm_name,
                                         self.NAME: p_instance_name, self.OWNER_KEY: p_user_key}

        persistance_manager.add_instance_to_database(key_generated, self.instances[key_generated])

        self.privilege_manager.register_instance(p_user_key, key_generated)

        return key_generated

    # Prise de décisions
    def get_decision(self, p_instance_key, p_context = None, p_filtre = None):

        self.check_instance_key(p_instance_key)

        if p_context is None:

            assert self.instances[p_instance_key][
                       self.TYPE] == TYPE_BANDIT, "Pour les algorithmes de bandits contextuels, inclure le contexte."

            return self.instances[p_instance_key][self.INSTANCE].select_action(p_filtre = p_filtre)

        else:

            assert self.instances[p_instance_key][
                       self.TYPE] == TYPE_BANDIT_CONTEXTUEL, "Ne pas inclure de contexte pour les algorithmes de bandits non contextuels"

            return self.instances[p_instance_key][self.INSTANCE].select_action(p_context, p_filtre)

    def observe(self, p_instance_key, p_action, p_reward, p_context=None):

        self.check_instance_key(p_instance_key)

        # Bandit non contextuel
        if p_context is None:
            self.instances[p_instance_key][self.INSTANCE].observe(p_action, p_reward)

        else:

            self.instances[p_instance_key][self.INSTANCE].observe(p_context, p_action, p_reward)

        persistance_manager.update_instance_in_database(p_instance_key, self.instances[p_instance_key])

    def reset(self):
        assert False, "Not implemented"

    def check_instance_key(self, p_instance_key):

        if p_instance_key not in self.instances:
            raise PrivilegeException("La clé instance n'existe pas.")
