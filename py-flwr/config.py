import os
import logging
import torch


class Config:
    # Hyperparameters

    n_features = 10
    n_outputs = 2
    hidden_size = 20

    batch_size = 32

    learning_rate = 0.001
    n_epochs = 50
    step_size = 25
    gamma = 0.1

    # directory to save the model's weight
    model_out = "model"
    best_model_name = 'best_model.pth'

    # --------------------
    # Config ClientApp
    # --------------------
    n_clients = 4
    n_client_epochs = 50

    # --------------------
    # Config Strategies
    # --------------------
    # Strategies: fedavg, fedavgm, fedmedian
    strategy_name = "fedavg"

    # --------------------
    # Config ServerApp
    # --------------------
    num_server_rounds = 10
    fraction_evaluate = 1
    test_fraction_size = 0.2

    # Config Strategy for FedAvgM
    server_momentum = 0.2
    server_learning_rate = 0.001

    # server address
    server_hostname = 'localhost'
    server_port = 8080

    # --------------------
    # Differential Privacy
    # --------------------
    # whether to use differential privacy
    enable_differential_privacy: bool = False
    # server side
    noise_multiplier:float = 0.25  # The noise multiplier for the Gaussian mechanism for model updates. A value of 1.0 or higher is recommended for strong privacy.
    clipping_norm: float = 20.0  # The value of the clipping norm.
    # client side
    sensitivity = 1.0
    epsilon = 0.1
    delta = 1e-5

    def __init__(self, *args, **kwargs):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        assert self.strategy_name in ['fedavg', 'fedavgm', 'fedmedian']
        assert 0 <= self.test_fraction_size < 1, "Must be in the range [0, 1)"
        logging.info(f"Using device: {self.device}")
        logging.info(f"Using strategy: {self.strategy_name}")
        logging.info(f"Using differential privacy: {self.enable_differential_privacy}")


os.environ["RAY_DEDUP_LOGS"] = "1"
logging.basicConfig(level=logging.INFO)
configs = Config(
    strategy_name="fedavg",
    enable_differential_privacy=True,
)
