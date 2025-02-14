
# Installation


```
cd fedl/
pyenv virtualenv 3.11 fedl
pyenv activate fedl
pip3 install -r requirements.txt
```

The following command can be used to verify if Flower was successfully installed.
If everything worked, it should print the version of Flower to the command line:

```
python -c "import flwr;print(flwr.__version__)"
```

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
