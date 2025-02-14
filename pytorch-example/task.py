"""pytorchexample: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import MyRegressionDataset
from config import N_FEATURES, N_OUTPUTS, TEST_SIZE
from config import BATCH_SIZE, NUM_CLIENTS


fds = None  # Cache FederatedDataset


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition mydataset data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        print("Initializing FederatedDataset")
        fds = MyRegressionDataset(N_FEATURES, N_OUTPUTS, num_partitions=num_partitions)

    partition = fds.create_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_data = partition.train_test_split(test_size=TEST_SIZE, seed=42)
    """
    partition_data is a DatasetDict containing the train and test partitions.
    for example if the dataset has 2500 samples, partition_data will contain:

    DatasetDict({
        train: Dataset({
            features: ['X', 'y'],
            num_rows: 2000  # 80% of 2500
        })
        test: Dataset({
            features: ['X', 'y'],
            num_rows: 500  # 20% of 2500
        })
    })
    """


    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        # batch.keys() ==> dict_keys(['X', 'y'])
        X = [torch.tensor(x) for x in batch["X"]]
        y = [torch.tensor(y) for y in batch["y"]]
        return {"X": X, "y": y}

    partition_data = partition_data.with_transform(apply_transforms)

    trainloader = DataLoader(
        partition_data["train"],
        batch_size=batch_size,
        shuffle=True
    )
    testloader = DataLoader(
        partition_data["test"],
        batch_size=batch_size,
        shuffle=False
    )
    return trainloader, testloader


def train(net, trainloader, valloader, epochs, learning_rate, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=learning_rate,
        momentum=0.9
    )
    net.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        total_elem = 0
        for batch in trainloader:
            X, y = batch["X"].to(device), batch["y"].to(device)
            optimizer.zero_grad()
            outputs = net(X)

            # has_nan = torch.isnan(outputs).any()
            # if has_nan:
            #     print(X)
            #     print(y)
            #     print(outputs)
            #     raise ValueError("outputs has NaNs in epoch", epoch)

            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            total_elem += len(y)
        epoch_loss /= total_elem
        # print(f"Epoch #{epoch} - loss: {epoch_loss:.4f}")

    val_loss, val_mae = test(net, valloader, device)

    results = {
        "val_loss": val_loss,
        "val_mae": val_mae,
    }
    return results


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)  # move model to GPU if available
    criterion = nn.MSELoss()
    acc_abs = 0.0  # accumulates the absolute error
    acc_loss = 0.0  # accumulates the loss
    with torch.no_grad():
        for batch in testloader:
            X, y = batch['X'].to(device), batch['y'].to(device)
            outputs = net(X)
            loss = criterion(outputs, y).item()
            acc_loss += loss
            acc_abs += torch.abs(outputs -  y).sum().item()
    # print("acc_abs", acc_abs, "len(testloader.dataset)", len(testloader.dataset))
    # print("acc_loss", acc_loss, "len(testloader)", len(testloader))
    val_mae = acc_abs / len(testloader.dataset)
    val_loss = acc_loss / len(testloader)
    return val_loss, val_mae


def test_load_partition():
    global fds
    fds = None  # guarantee that the dataset is re-initialized

    partition_id = 0
    num_partitions = NUM_CLIENTS
    print("Calling load_data with partition_id", partition_id)
    trainloader, testloader = load_data(partition_id, num_partitions, BATCH_SIZE)
    data = next(iter(trainloader))
    # print(data.keys())
    # print("Number of samples:", len(data['X']), data['X'].shape)
    print("Data - X:", data['X'].shape, 'Y:', data['y'].shape)

    assert len(data['X']) == BATCH_SIZE and len(data['y']) == BATCH_SIZE
    assert data['X'].shape[1] == N_FEATURES
    assert data['y'].shape[1] == N_OUTPUTS


if __name__ == "__main__":
    import argparse
    from mymodel import RegressionModel

    parser = argparse.ArgumentParser(description='Federated learning example')
    parser.add_argument('--test_load', dest='test_load', action='store_true', help='test loading data')
    parser.add_argument('--no_test_load', dest='test_load', action='store_false')
    parser.add_argument('--train', dest='train', action='store_true', help='train a model')
    parser.add_argument('--no_train', dest='train', action='store_false')
    parser.set_defaults(test_load=False, train=True)
    args = parser.parse_args()

    if args.test_load:
        test_load_partition()

    if args.train:
        dataset = MyRegressionDataset(N_FEATURES, N_OUTPUTS)
        partition_id = 0
        num_partitions = NUM_CLIENTS
        trainloader, testloader = load_data(partition_id, num_partitions, BATCH_SIZE)

        model = RegressionModel(N_FEATURES, N_OUTPUTS)

        result = train(model,
            trainloader, testloader,
            epochs=50,
            learning_rate=0.0001,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        print("Test:", result)
