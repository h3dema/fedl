# Example of homomorphic encryption

This example focuses on the core ideas and omits some implementation details for clarity.
It uses a simplified version of HE for demonstration.

Scenario:
<pre>
We want to train a simple model (e.g., linear regression) on data held by multiple clients, but we don't want the clients to share their raw data or model updates directly with the server.
</pre>


## Simplified HE Scheme (for illustration):

Imagine a very simplified HE scheme where:

> Encryption: $Enc(x) = x + k$

where k is a secret key unique to each client

> Decryption: $Dec(y) = y - k$

> Homomorphic Addition: $Enc(x) + Enc(y) = (x + k_1) + (y + k_2) = (x + y) + (k_1 + k_2)$

We'll assume the server can somehow handle the combined keys or that each client uses the same key for this simplified example.


## Flower Integration (Conceptual):

### Client-Side Training and Encryption:

Each client trains a local model on its data.
Instead of sending the model updates (e.g., gradients or model weights) directly to the server, the client encrypts them using its HE public key.

```Python
import flwr as fl
import numpy as np  # For simplicity

class Client(fl.client.NumPyClient):
    def get_parameters(self, config):
        # ... (get local model parameters) ...
        return parameters

    def fit(self, parameters, config):
        # ... (local training) ...
        updates = calculate_model_updates(local_model, local_data) # e.g., gradients

        # Encrypt the updates (simplified example)
        encrypted_updates = [update + client_secret_key for update in updates] # Simplified encryption

        return encrypted_updates, num_examples, {}  # Return encrypted updates

    def evaluate(self, parameters, config):
        # ... (local evaluation) ...
        return loss, accuracy, num_examples

fl.client.start_numpy_client(client=Client())
```

### Server-Side Aggregation:

The server receives the encrypted model updates from all clients.
The server performs homomorphic addition on the encrypted updates. Because of the homomorphic property, this is equivalent to adding the unencrypted updates.

```Python
def aggregate_fit(updates, config):
    # updates is a list of (encrypted_updates, num_examples) tuples
    encrypted_aggregated_updates = sum([updates[i][0] for i in range(len(updates))]) # Simplified homomorphic addition

    # ... (Server-side differential privacy could be applied here) ...

    # Server DOES NOT decrypt! It sends the encrypted aggregated update back to a client (or distributes it).
    return encrypted_aggregated_updates, {} # aggregated updates, metrics

strategy = fl.server.strategies.FedAvg(
    aggregate_fn=aggregate_fit,
)
fl.server.start_server(strategy=strategy)
```


### Client-Side Decryption and Model Update:

One (or more) designated client(s) receives the encrypted aggregated model update from the server.
The client decrypts the aggregated update using its HE private key.
The client applies the decrypted update to its local model.

```Python
# ... (In the Client class, after receiving the global model) ...

def update_model(self, encrypted_aggregated_updates):
    # Decrypt (simplified)
    aggregated_updates = [update - client_secret_key for update in encrypted_aggregated_updates]  # Simplified decryption

    # Apply to local model
    apply_updates(local_model, aggregated_updates)  # Update local model
```


### Key Improvements and Real-World Considerations:

- **Real HE Schemes**: Instead of the simplified example, you'd use a real HE scheme like BFV or CKKS from libraries like SEAL or PALISADE. This would require more sophisticated encryption and homomorphic operations.
- **Secure Aggregation**: More advanced secure aggregation techniques would be needed to ensure that the server doesn't learn anything about the individual client updates, even with HE.
- **Differential Privacy**: Often, HE is combined with differential privacy to further enhance privacy. Noise would be added to the model updates or the aggregated model.
- **Computational Cost**: HE is computationally expensive. You'd need to carefully consider the trade-off between privacy and performance.

Implementing HE in FL with Flower would involve significant complexity, particularly in managing keys, choosing appropriate HE schemes, and handling the computational overhead.
However, libraries like TenSEAL can make this process somewhat easier.