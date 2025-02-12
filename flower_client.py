import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import flwr as fl

from config import batch_size

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, testset):
        self.testset = testset
        self.model = model
        self.train_loader = train_loader
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

    def get_parameters(self, config):
        return [param.cpu().numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):  # Train for 1 epoch
            for images, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

        # returns parameters, num_examples, metrics (dict)
        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        mse, total = 0, 0
        with torch.no_grad():
            for images, labels in torch.utils.data.DataLoader(self.testset, batch_size=batch_size):
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                mse += torch.sum((predicted - labels) ** 2).item()

        mse /= total

        # returns loss, num_examples, metrics (dict)
        return float(mse.item()), len(self.testset), {"mse": mse}
