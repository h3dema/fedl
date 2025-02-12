import flwr as fl
from typing import Dict

from config import HOSTNAME, PORT


def weighted_average(metrics: Dict):
    """Aggregation function for metrics."""
    mse = [num_examples * m["mse"] for num_examples, m in zip(metrics["num_examples"], metrics["metrics"])]
    return {"mse": sum(mse) / sum(metrics["num_examples"])}


# Start the FL server
if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)
    url = f"{HOSTNAME}:{PORT}"
    fl.server.start_server(
        server_address=url,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )
