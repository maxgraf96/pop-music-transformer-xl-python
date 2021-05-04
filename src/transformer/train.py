# =============================================
# Default code to train the model. Not used locally.
# This code was copied to a python notebook on the compute servers to train the model
# =============================================

import os
import numpy as np
import torch

from tranformerxl import configure, train
from constants import MODEL_PATH
np.seterr('raise')

SEED = 987654321
input_length = 512
n_token = 308

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path_chkp = MODEL_PATH if os.path.exists(MODEL_PATH) else None
path_chkp = None

# Configure model
model = configure()

# Train
train(model, num_epochs=100, num_segments=800)