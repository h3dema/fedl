# Hyperparameters

N_FEATURES = 10
N_OUTPUTS = 2
HIDDEN_SIZE = 20

BATCH_SIZE = 32
LOG_INTERVAL = 10

LR = 0.001
n_epochs = 50
step_size = 25
gamma = 0.1

DEVICE = "cpu"

TEST_SIZE = 0.2

assert 0 <= TEST_SIZE < 1, "TEST_SIZE should be a fraction"