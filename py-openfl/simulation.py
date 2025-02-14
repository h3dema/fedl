import numpy as np
from copy import deepcopy
import torch
from torch import optim
import torch.nn.functional as F

"""
openfl '1.7'
metaflow '2.14.0'

1) ImportError: cannot import name 'DATASTORES' from 'metaflow.datastore'

- on openfl/experimental/workflow/utilities/metaflow_utils.py.
- check `metaflow_utils.diff` for the changes.

2) ImportError: cannot import name 'StepVisitor' from 'metaflow.graph'
- this class does not exist

"""

from openfl.experimental.workflow.interface import FLSpec, Aggregator, Collaborator
from openfl.experimental.workflow.runtime import LocalRuntime
from openfl.experimental.workflow.placement import aggregator, collaborator

from config import N_FEATURES, N_OUTPUTS
from config import n_epochs, LR, LOG_INTERVAL
from config import DEVICE
from config import TEST_SIZE, BATCH_SIZE
from dataset import MyRegressionDataset
from mymodel import RegressionModel


MOMENTUM = 0.5


def FedAvg(models, weights=None):
    new_model = models[0]
    state_dicts = [model.state_dict() for model in models]
    state_dict = new_model.state_dict()
    for key in models[1].state_dict():
        state_dict[key] = torch.from_numpy(
            np.average([state[key].numpy() for state in state_dicts],
                       axis=0,
                       weights=weights,
            )
        )
    new_model.load_state_dict(state_dict)
    return new_model


def inference(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
      for data, target in test_loader:
        output = model(data)
        test_loss += F.mse_loss(output, target, size_average=False).item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}'.format(test_loss))
    return test_loss


class FederatedFlow(FLSpec):

    def __init__(self, model=None, optimizer=None, rounds=3, **kwargs):
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            self.optimizer = optimizer
        else:
            self.model = RegressionModel(N_FEATURES, N_OUTPUTS)
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=LR,
                momentum=MOMENTUM,
            )
        self.rounds = rounds

    @aggregator
    def start(self):
        print(f'Performing initialization for model')
        self.collaborators = self.runtime.collaborators
        self.private = 10
        self.current_round = 0
        self.next(self.aggregated_model_validation, foreach='collaborators', exclude=['private'])

    @collaborator
    def aggregated_model_validation(self):
        print(f'Performing aggregated model validation for collaborator {self.input}')
        self.agg_validation_score = inference(self.model, self.test_loader)
        print(f'{self.input} value of {self.agg_validation_score}')
        self.next(self.train)

    @collaborator
    def train(self):
        self.model.train()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=LR,
            momentum=MOMENTUM,
        )

        best_loss = None
        self.loss = 0  # this value is exported to the aggregator
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()

            # print training stats
            if batch_idx % LOG_INTERVAL == 0:
                print('Train Epoch: 1 [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))
                self.loss += loss.item()  # save accumulated training loss

            # save best model
            if best_loss is None or loss < best_loss:
                best_loss = loss
                torch.save(self.model.state_dict(), 'model.pth')
                torch.save(self.optimizer.state_dict(), 'optimizer.pth')
        self.training_completed = True
        self.next(self.local_model_validation)

    @collaborator
    def local_model_validation(self):
        self.local_validation_score = inference(self.model, self.test_loader)
        print(f'Doing local model validation for collaborator {self.input}: {self.local_validation_score}')
        self.next(self.join, exclude=['training_completed'])

    @aggregator
    def join(self, inputs):
        self.average_loss = sum(input.loss for input in inputs) / len(inputs)
        self.aggregated_model_loss = sum(input.local_validation_score for input in inputs) / len(inputs)
        print(f'Average training loss = {self.average_loss}')
        print(f'Average aggregated model validation values = {self.aggregated_model_loss}')
        self.model = FedAvg([input.model for input in inputs])
        self.optimizer = [input.optimizer for input in inputs][0]
        self.current_round += 1
        if self.current_round < self.rounds:
            self.next(self.aggregated_model_validation,
                      foreach='collaborators', exclude=['private'])
        else:
            self.next(self.end)

    @aggregator
    def end(self):
        print(f'This is the end of the flow')


if __name__ == "__main__":
    regr_dataset = MyRegressionDataset(N_FEATURES, N_OUTPUTS)
    train_dataset, test_dataset = torch.utils.data.random_split(regr_dataset, [1 - TEST_SIZE, TEST_SIZE])

    model = RegressionModel(N_FEATURES, N_OUTPUTS)
    optimizer = optim.SGD(model.parameters(), lr=LR)

    # Setup participants
    aggregator = Aggregator()
    aggregator.private_attributes = {}

    # Setup collaborators with private attributes
    collaborator_names = ['Portland', 'Seattle', 'Chandler','Bangalore']
    collaborators = [Collaborator(name=name) for name in collaborator_names]
    for idx, collaborator in enumerate(collaborators):

        local_train = deepcopy(train_dataset)
        local_train.data = train_dataset.data[idx::len(collaborators)]
        local_train.targets = train_dataset.targets[idx::len(collaborators)]

        local_test = deepcopy(test_dataset)
        local_test.data = test_dataset.data[idx::len(collaborators)]
        local_test.targets = test_dataset.targets[idx::len(collaborators)]

        collaborator.private_attributes = {
                'train_loader': torch.utils.data.DataLoader(local_train,batch_size=BATCH_SIZE, shuffle=True),
                'test_loader': torch.utils.data.DataLoader(local_test,batch_size=BATCH_SIZE, shuffle=True)
        }

    local_runtime = LocalRuntime(aggregator=aggregator, collaborators=collaborators, backend='single_process')
    print(f'Local runtime collaborators = {local_runtime.collaborators}')

    # run the experiment
    model = None
    best_model = None
    optimizer = None
    flflow = FederatedFlow(model, optimizer, rounds=2, checkpoint=True)
    flflow.runtime = local_runtime
    flflow.run()

    print(f'The final model weights: {flflow.model.state_dict()}')
    print(f'\nFinal aggregated model accuracy for {flflow.rounds} rounds of training: {flflow.aggregated_model_loss}')