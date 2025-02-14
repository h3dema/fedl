import multiprocessing
import flwr as fl

from config import n_features, n_outputs
from config import HOSTNAME, PORT
from dataset import MyRegressionDataset, get_client_data
from model import RegressionModel
from flower_client import FlowerClient


def start_client(trainset, testset, client_id: int):
    model = RegressionModel(n_features=n_features, n_outputs=n_outputs)
    train_loader = get_client_data(trainset, client_id)
    client = FlowerClient(model, train_loader, testset)
    url = f"{HOSTNAME}:{PORT}"
    fl.client.start_client(server_address=url, client=client)


# Start two clients
if __name__ == "__main__":

    trainset = MyRegressionDataset(n_features, n_outputs)
    testset = MyRegressionDataset(n_features, n_outputs, n_samples=50, random_state=20)

    num_clients = 2
    processes = []
    for i in range(num_clients):
        p = multiprocessing.Process(
            target=start_client,
            args=(trainset, testset, i,)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
