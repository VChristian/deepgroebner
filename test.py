import tensorflow as tf
import os
import datetime
import numpy as np
from deepgroebner.ac import Agent_AC
from deepgroebner.tpmlp_mha_scoring import TransformerPMLP_Score_MHA
from deepgroebner.tpmlp_q_scoring import TransformerLayer_Score_Q
from deepgroebner.wrapped import CLeadMonomialsEnv

env = CLeadMonomialsEnv('3-20-10-weighted', elimination='gebauermoeller', rewards='additions', k=2)
tpmlp_scoring = TransformerPMLP_Score_MHA(score_layers = [128, 64], dim = 128, hidden_dim = 128)

agent = Agent_AC(tpmlp_scoring)
agent.train(env, epochs = 20, episodes = 20, verbose=2, logdir = os.path.join('test', 'runs'), parallel=False)