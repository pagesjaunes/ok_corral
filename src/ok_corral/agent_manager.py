from ok_corral.bandits import *

#Type d'algorithmes

TYPE_BANDIT = 0

class AgentManager:

    def __init__(self):

        self.agents = {}

    def add_bandit(self, p_instance_name, p_algorithm_name, p_nombre_bras):

        key_generated = "42"

        # Vérification validité de la clé

        if p_algorithm_name == THOMPSON_SAMPING:

            self.agents[key_generated] = {"instance" : ThompsonSampling(p_nombre_bras), "type" : TYPE_BANDIT, "name" : p_instance_name}

        if p_algorithm_name == UPPER_CONFIDENCE_BOUND:

            self.agents[key_generated] = {"instance" : UCB(p_nombre_bras), "type" : TYPE_BANDIT, "name" : p_instance_name}

        else:

            self.agents[key_generated] = {"instance" : RandomBandit(p_nombre_bras), "type" : TYPE_BANDIT, "name" : p_instance_name}

        return {"key" : key_generated}


    def get_decision(self,p_key):

        return {"action" : self.agents[p_key]["instance"].select_action()}


    def observe(self, p_key, p_action, p_reward):

        self.agents[p_key]["instance"].observe(self,p_action, p_reward)


    def reset(self):
        pass