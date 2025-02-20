import torch
from flwr.server import ServerApp
from flwr.client import ClientApp
from flwr.simulation import run_simulation
from flwr.client.mod import LocalDpMod

from server_app import server_fn
from client_app import client_fn

from config import configs


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

    # Differential Privacy
    # ref. https://flower.ai/docs/framework/ref-api/flwr.client.mod.LocalDpMod.html#flwr.client.mod.LocalDpMod
    mods = []
    if configs.enable_differential_privacy:
        local_dp_obj = LocalDpMod(configs.clipping_norm, configs.sensitivity, configs.epsilon, configs.delta)
        mods = [local_dp_obj]

    # Flower ClientApp
    client = ClientApp(
        client_fn,
        mods=mods,
        )

    # Specify the resources each of your clients need
    # By default, each client will be allocated 1x CPU and 0x GPUs
    backend_config = {
        "client_resources": {
            "num_cpus": 1,
            "num_gpus": 0.0,
        }
    }

    # When running on GPU, assign an entire GPU for each client
    if configs.device == "cuda" and torch.cuda.device_count() >= configs.n_clients:
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
        num_supernodes=configs.n_clients,
        backend_config=backend_config,
    )
