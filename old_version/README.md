This folder uses deprecated calls to Flower.ai toolkit.

# How to Run the Example

1. Start the server first:
```sh
python server.py
```

2. Start the clients (each in a separate terminal or process):
```sh
python clients.py
```

The clients will train locally and send updates to the server.
This example uses Federated Averaging (FedAvg) to aggregate model updates from different clients.
You can modify the number of clients, dataset partitioning, and model architecture to experiment further.