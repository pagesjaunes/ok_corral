RANDOM_BANDIT = "random"
THOMPSON_SAMPING = "ts"
UPPER_CONFIDENCE_BOUND = "ucb"
LINUCB = "linucb"

BANDIT_AVAILABLES = [THOMPSON_SAMPING, UPPER_CONFIDENCE_BOUND, LINUCB]

from ok_corral.engine.agents import LinUCB
from ok_corral.engine.agents import RandomContextualBandit

from ok_corral.engine.agents import RandomBandit
from ok_corral.engine.agents import ThompsonSampling
from ok_corral.engine.agents import UCB


def get_class_from_algo(p_type):

    if p_type == RANDOM_BANDIT:
        return RandomBandit
    elif p_type == THOMPSON_SAMPING:
        return ThompsonSampling
    elif p_type == UPPER_CONFIDENCE_BOUND:
        return UCB
    elif p_type == LINUCB:
        return LinUCB
    else:
        assert False