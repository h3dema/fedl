import random
from sklearn.datasets import make_regression

import torch
from torch.utils.data import Dataset as TorchDataset

from datasets import Dataset
from flwr_datasets.partitioner import IidPartitioner

from config import BATCH_SIZE


def gen_dict_dataset(input_dim, output_dim, n_samples=5000, noise=0.5, random_state=42):
    """
    Generate a synthetic regression dataset and convert it to a PyTorch Dataset.

    Parameters:
    input_dim (int): The number of features for each sample.
    output_dim (int): The number of target outputs for each sample.
    n_samples (int, optional): The total number of samples to generate. Default is 5000.
    noise (float, optional): The standard deviation of the gaussian noise added to the output. Default is 0.5.
    random_state (int, optional): Determines random number generation for dataset creation. Default is 42.

    Returns:
    Dataset: A PyTorch Dataset containing the generated samples and targets.
    """

    X, y = make_regression(
        n_samples=n_samples,
        n_features=input_dim,
        n_targets=output_dim,
        noise=noise,
        random_state=random_state,
    )
    data = dict(
        X = torch.tensor(X).float(),
        y = torch.tensor(y).float()
    )
    dict_dataset = Dataset.from_dict(data)

    return dict_dataset


class MyRegressionDataset(TorchDataset):

    def __init__(self, n_features, n_outputs, num_partitions=1):
        """
        Initialize the MyRegressionDataset class.

        Args:
            n_features (int): The number of features in the dataset.
            n_outputs (int): The number of outputs in the dataset.
            num_partitions (int): The number of partitions to split the dataset into.
        """
        super().__init__()
        self.n_features = n_features
        self.n_outputs = n_outputs

        self.partitioner = IidPartitioner(num_partitions=num_partitions)
        self.dataset = gen_dict_dataset(n_features, n_outputs)
        self.partitioner.dataset = self.dataset

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.dataset["X"])

    def __getitem__(self, idx) -> tuple:
        """
        Return the sample at the given index.

        Args:
            idx (int): The index of the sample to be retrieved.

        Returns:
            tuple: A tuple containing the input and output of the sample at the given index.
        """

        return torch.tensor(self.dataset["X"][idx]), torch.tensor(self.dataset["y"][idx])

    def create_partition(self, partition_id) -> Dataset:
        """
        Create and return a partition of the dataset using the specified partition ID.

        Args:
            partition_id (int): The ID of the partition to be created.

        Returns:
            Dataset: A partition of the dataset corresponding to the given partition ID.
        """

        partition = self.partitioner.load_partition(partition_id)
        return partition


if __name__ == "__main__":
    from config import N_FEATURES, N_OUTPUTS
    dataset = MyRegressionDataset(N_FEATURES, N_OUTPUTS, num_partitions=2)
    data = dataset[0]
    print("X", data[0].shape, "y", data[1].shape)

    partition = dataset.create_partition(0)
    print(partition)
    """ returns:
            Dataset({
                features: ['X', 'y'],
                num_rows: 2500
            })

        partition['X'] is a list of 2500 samples
        partition['X'][0] is a list of `input_dim` features from sample #0, etc.
    """
