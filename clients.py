import multiprocessing
import flwr as fl

from config import input_dim, output_dim
from config import HOSTNAME, PORT
from dataset import MyRegressionDataset, get_client_data
from model import RegressionModel
from flower_client import FlowerClient


def start_client(trainset, testset, client_id: int):
    model = RegressionModel(n_features=input_dim, n_outputs=output_dim)
    train_loader = get_client_data(trainset, client_id)
    client = FlowerClient(model, train_loader, testset)
    url = f"{HOSTNAME}:{PORT}"
    fl.client.start_numpy_client(server_address=url, client=client)


# Start two clients
if __name__ == "__main__":

    trainset = MyRegressionDataset(input_dim, output_dim)
    testset = MyRegressionDataset(input_dim, output_dim, n_samples=50, random_state=20)

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
