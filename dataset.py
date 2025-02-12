import random
from sklearn.datasets import make_regression

import torch
from torch.utils.data import Dataset


class MyRegressionDataset(Dataset):
    """
        MyRegressionDataset is a custom dataset class for generating a synthetic regression dataset using PyTorch and scikit-learn.

    """

    def __init__(self, input_dim, output_dim, n_samples=5000, random_state=42):
        """
        Initializes the dataset with input and output dimensions and generates a regression dataset.

        Args:
            input_dim (int): The number of input features.
            output_dim (int): The number of output targets.
            n_samples (int, optional): The number of samples to generate. Default is 5000.

        Attributes:
            X (torch.Tensor): The input features tensor.
            y (torch.Tensor): The output targets tensor.
        """
        super().__init__()
        self.X, self.y = make_regression(
            n_samples=n_samples,
            n_features=input_dim,
            n_targets=output_dim,
            noise=0.5,
            random_state=random_state,
        )
        self.X = torch.tensor(self.X).float()
        self.y = torch.tensor(self.y).float()

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return self.X.shape[0]

    def __getitem__(self, index):
        """
        Retrieve the data sample and corresponding label at the specified index.

        Args:
            index (int): The index of the data sample to retrieve.

        Returns:
            tuple: A tuple containing the data sample (self.X[index]) and the corresponding label (self.y[index]).
        """
        return self.X[index], self.y[index]


# Split dataset into multiple parts for clients
def get_client_data(trainset: Dataset, client_id: int, num_clients: int = 2):
    """
    Splits the dataset for different clients.

    Args:
        trainset (Dataset): The dataset to be split.
        client_id (int): The id of the client.
        num_clients (int, optional): The number of clients. Defaults to 2.

    Returns:
        DataLoader: A DataLoader containing the split dataset for the specified client.
    """
    total_size = len(trainset)
    indices = list(range(total_size))
    random.shuffle(indices)
    part_size = total_size // num_clients
    start_idx = client_id * part_size
    end_idx = start_idx + part_size
    subset = torch.utils.data.Subset(trainset, indices[start_idx:end_idx])
    return torch.utils.data.DataLoader(subset, batch_size=32, shuffle=True)


if __name__ == "__main__":
    input_dim, output_dim = 10, 2
    data = MyRegressionDataset(input_dim, output_dim)

    x, y = data[0]
    print("x:", x)
    print("y:", y)
