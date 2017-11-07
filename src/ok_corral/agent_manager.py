import os
from binascii import hexlify

from ok_corral.bandits import *
#Type d'algorithmes

TYPE_BANDIT = "bandit"
TYPE_BANDIT_CONTEXTUEL = "bandit_contextuel"


def _key_generation(p_agent):

    key = hexlify(os.urandom(256)).decode()

    while key in p_agent.agents:
        key = hexlify(os.urandom(256)).decode()

    return key


class PrivilegeException(Exception):

    pass

class PrivilegeManager:

    def __init__(self):

        self.user_keys = set()
        self.user_owned = {}

        self.user_keys.add("skeleton_key_42")

    def can_create(self, p_user_key):

        if p_user_key in self.user_keys:

            return True
        else:
            return False

    def register_instance(self,p_user_key, p_instance_key):

        if p_user_key not in self.user_owned:
            self.user_owned[p_user_key] = []

        self.user_owned[p_user_key].append(p_instance_key)


class AgentManager:

    def __init__(self):

        self.agents = {}
        self.privilege_manager = PrivilegeManager()

    # Gestion d'instance

    def add_bandit(self, p_user_key, p_instance_name, p_algorithm_name, p_nombre_bras, p_context_description = None):

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

        self.agents[key_generated] = {"instance" : instance, "type" : type, "algorithme": p_algorithm_name, "name" : p_instance_name, "owner_key" : p_user_key}

        self.privilege_manager.register_instance(p_user_key,key_generated)

        return key_generated

    # Prise de décisions

    def get_decision(self,p_instance_key, p_context = None):

        self.check_instance_key(p_instance_key)

        if p_context is None:

            assert self.agents[p_instance_key]["type"] == TYPE_BANDIT, "Pour les algorithmes de bandits contextuels, inclure le contexte."
            return self.agents[p_instance_key]["instance"].select_action()

        else:

            assert self.agents[p_instance_key]["type"] == TYPE_BANDIT_CONTEXTUEL, "Ne pas inclure de contexte pour les algorithmes de bandits non contextuels"
            return self.agents[p_instance_key]["instance"].select_action(p_context)


    def observe(self, p_instance_key, p_action, p_reward, p_context = None):

        self.check_instance_key(p_instance_key)

        # Bandit non contextuel
        if p_context is None:
            self.agents[p_instance_key]["instance"].observe(p_action, p_reward)

        else:

            print(p_instance_key)
            self.agents[p_instance_key]["instance"].observe(p_context, p_action, p_reward)

    def reset(self):
        assert False, "Not implemented"


    def check_instance_key(self, p_instance_key):

        if p_instance_key not in self.agents:
            raise PrivilegeException("La clé instance n'existe pas.")
