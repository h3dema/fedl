import torch
from flwr.server import ServerApp
from flwr.client import ClientApp
from flwr.simulation import run_simulation

from server_app import server_fn
from client_app import client_fn

from config import DEVICE, NUM_CLIENTS

# Example:
# --------
#
#  python3 simulation.py
#
if __name__ == "__main__":
    # Create ServerApp
    server = ServerApp(
        server_fn=server_fn
    )

    # Flower ClientApp
    client = ClientApp(client_fn)

    # Specify the resources each of your clients need
    # By default, each client will be allocated 1x CPU and 0x GPUs
    backend_config = {
        "client_resources": {
            "num_cpus": 1,
            "num_gpus": 0.0,
        }
    }

    # When running on GPU, assign an entire GPU for each client
    if DEVICE == "cuda" and torch.cuda.device_count() >= NUM_CLIENTS:
        print("Using GPU backend")
        backend_config = {
            "client_resources": {
                "num_cpus": 1,
                "num_gpus": 1.0
            }
        }

    # Run simulation
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
    )