"""pytorchexample: A Flower / PyTorch app."""

from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import Driver
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedAvgM, FedMedian
from flwr.server.strategy import DifferentialPrivacyServerSideFixedClipping

from mymodel import RegressionModel
from task import get_weights
from config import configs


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics by weighted average.

    Args:
        metrics (List[Tuple[int, Metrics]]): List of tuples (num_examples, calc_metrics).
        len(metrics) is equal to num_clients.
        calc_metrics is a dictionary containing the all calculated metrics for the client (check evaluate() in FlowerClient).
        Example:
        [(500, {'val_mae': 101.09595703125}), (500, {'val_mae': 99.91997802734375})]
    """
    # Multiply MAE of each client by number of examples used
    val_mae = [num_examples * m["val_mae"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"val_mae": sum(val_mae) / sum(examples)}


def server_fn(context: Context):
    # print("SERVER:", context)
    """Construct components that set the ServerApp behaviour."""

    # Initialize model parameters
    ndarrays = get_weights(RegressionModel(configs.n_features, configs.n_outputs))
    parameters = ndarrays_to_parameters(ndarrays)

    # Define the strategy
    if configs.strategy_name == "fedavg":
        # Federated Averaging strategy.
        # Implementation based on https://arxiv.org/abs/1602.05629
        strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=configs.fraction_evaluate,
            min_available_clients=2,
            evaluate_metrics_aggregation_fn=weighted_average,
            initial_parameters=parameters,
        )

    elif configs.strategy_name == "fedavgm":
        # Federated Averaging with Momentum strategy.
        # Implementation based on https://arxiv.org/abs/1909.06335
        strategy = FedAvgM(
            fraction_fit=1.0,
            fraction_evaluate=configs.fraction_evaluate,
            min_available_clients=2,
            evaluate_metrics_aggregation_fn=weighted_average,
            initial_parameters=parameters,
            server_momentum=configs.server_momentum,
            server_learning_rate=configs.server_learning_rate,
        )

    elif configs.strategy_name == "fedmedian":
        # Federated Median strategy.
        strategy = FedMedian(
            fraction_fit=1.0,
            fraction_evaluate=configs.fraction_evaluate,
            min_available_clients=2,
            evaluate_metrics_aggregation_fn=weighted_average,
            initial_parameters=parameters,
        )

    else:
        raise Exception(f"Strategy {configs.strategy_name} not implemented.")

    # Differential Privacy
    # ref. https://flower.ai/docs/framework/ref-api/flwr.server.strategy.DifferentialPrivacyServerSideFixedClipping.html
    if configs.enable_differential_privacy:
        strategy = DifferentialPrivacyServerSideFixedClipping(
            strategy,
            configs.noise_multiplier,
            configs.clipping_norm,
            configs.n_clients,
        )

    config = ServerConfig(num_rounds=configs.num_server_rounds)

    return ServerAppComponents(strategy=strategy, config=config)
