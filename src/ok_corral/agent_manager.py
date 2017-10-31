from ok_corral.bandits import *
import os

#Type d'algorithmes

TYPE_BANDIT = 0

def _key_generation(p_agent):

    key = os.urandom(24).encode('hex')

    while key in p_agent.agents:
        key = os.urandom(24).encode('hex')

    return key


class PrivilegeManager:

    def __init__(self):

        self.user_keys = set()
        self.user_owned = {}

        self.admin_keys.add("skeleton_key_42")

    def can_create(self, p_user_key):

        if p_user_key in self.admin_keys:

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

    def add_bandit(self, p_user_key, p_instance_name, p_algorithm_name, p_nombre_bras):

        if not self.privilege_manager.can_create(p_user_key):

            return None

        key_generated = _key_generation(self)

        # Vérification validité de la clé


        instance = None
        type = None

        if p_algorithm_name == THOMPSON_SAMPING:

            instance = ThompsonSampling(p_nombre_bras)
            type = TYPE_BANDIT

        if p_algorithm_name == UPPER_CONFIDENCE_BOUND:
            instance = UCB(p_nombre_bras)
            type = TYPE_BANDIT

        else:

            instance = RandomBandit(p_nombre_bras)
            type = TYPE_BANDIT

        self.agents[key_generated] = {"instance" : instance, "type" : TYPE_BANDIT, "algorithme": p_algorithm_name, "name" : p_instance_name, "owner_key" : p_user_key}

        self.privilege_manager.register_instance(p_user_key,key_generated)

        return key_generated

    # Prise de décisions

    def get_decision(self,p_instance_key):

        return self.agents[p_instance_key]["instance"].select_action()


    def observe(self, p_instance_key, p_action, p_reward):

        self.agents[p_instance_key]["instance"].observe(self,p_action, p_reward)


    def reset(self):
        pass