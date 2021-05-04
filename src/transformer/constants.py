# =============================================
# Contains all constants used in the system
# as well as a dynamic model path
# =============================================

import os

calling_dir = os.getcwd()
prefix = '' if 'transformer' in calling_dir else 'transformer/'
MODEL_PATH = prefix + 'REMI-my-checkpoint/model.pt'

INPUT_LENGTH = 512