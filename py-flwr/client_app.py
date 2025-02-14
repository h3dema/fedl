"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.client import NumPyClient
from flwr.common import Context

from mymodel import RegressionModel
from task import get_weights, load_data, set_weights, test, train

from config import N_FEATURES, N_OUTPUTS
from config import BATCH_SIZE, LR
from config import n_client_epochs
from config import DEVICE


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, local_epochs, learning_rate):
        self.net = RegressionModel(N_FEATURES, N_OUTPUTS)
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = DEVICE

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_weights(self.net, parameters)
        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.lr,
            self.device,
        )
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.net, parameters)
        loss, val_mae = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"val_mae": val_mae}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Read run_config to fetch hyperparameters relevant to this run
    trainloader, valloader = load_data(partition_id, num_partitions, BATCH_SIZE)

    # Return Client instance
    return FlowerClient(trainloader, valloader, n_client_epochs, LR).to_client()
