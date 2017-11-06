RANDOM_BANDIT = "random"
THOMPSON_SAMPING = "ts"
UPPER_CONFIDENCE_BOUND = "ucb"
LINUCB = "linucb"

BANDIT_AVAILABLES = [THOMPSON_SAMPING,UPPER_CONFIDENCE_BOUND, LINUCB]

from ok_corral.agents.agents import ThompsonSampling
from ok_corral.agents.agents import RandomBandit
from ok_corral.agents.agents import UCB

from ok_corral.agents.agents import LinUCB
from ok_corral.agents.agents import RandomContextualBandit

from ok_corral.feature_wrapper.feature_wrapper import FeatureWrapper