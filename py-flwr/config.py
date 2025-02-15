import torch

# Hyperparameters

N_FEATURES = 10
N_OUTPUTS = 2
HIDDEN_SIZE = 20

BATCH_SIZE = 32

LR = 0.001
n_epochs = 50
step_size = 25
gamma = 0.1


# Configuration

# directory to save the model's weight
model_out = "model"
best_model_name = 'best_model.pth'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

# Config ClientApp
NUM_CLIENTS = 4
n_client_epochs = 50

# Config ServerApp
num_server_rounds = 10
fraction_evaluate = 1
TEST_SIZE = 0.2

assert 0 <= TEST_SIZE < 1

# server address
HOSTNAME = 'localhost'
PORT = 8080
