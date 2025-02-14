import torch

# Hyperparameters

N_FEATURES = 10
N_OUTPUTS = 2
HIDDEN_SIZE = 20

BATCH_SIZE = 32

LR = 0.001
n_epochs = 100
step_size = 25
gamma = 0.1


# Configuration

# directory to save the model's weight
model_out = "model"
best_model_name = 'best_model.pth'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Config ClientApp
NUM_CLIENTS = 2
n_client_epochs = 20

# Config ServerApp
num_server_rounds = 2
fraction_evaluate = 1
TEST_SIZE = 0.2

# server address
HOSTNAME = 'localhost'
PORT = 8080
