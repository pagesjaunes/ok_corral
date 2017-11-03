RANDOM_BANDIT = "random"
THOMPSON_SAMPING = "ts"
UPPER_CONFIDENCE_BOUND = "ucb"

BANDIT_AVAILABLES = [THOMPSON_SAMPING,UPPER_CONFIDENCE_BOUND]

from ok_corral.agents.agents import ThompsonSampling
from ok_corral.agents.agents import RandomBandit
from ok_corral.agents.agents import UCB

from ok_corral.agents.agents import LinUCB
from ok_corral.agents.agents import RandomContextualBandit