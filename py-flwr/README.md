# Federated Learning with PyTorch and Flower


## Install

```bash
cd py-flwr
pyenv virtualenv 3.11 flwr
pyenv activate flwr
pip install -r requirements.txt
```

## Run the project

This example might run faster when the `ClientApp`s have access to a GPU.
If your system has one, you can make use of it by configuring the `DEVICE` in `config.py`.


```bash
pyenv activate flwr
cd py-flwr
python3 simulation.py
```

You can change the configuration in [`config.py`](config.py).

# Aggregation Strategies

Federated learning (FL) is a distributed learning paradigm where multiple clients collaboratively train a model without sharing their raw data.
The clients share their model updates with the server, which then aggregates them to create a new global model.
Several aggregation methods exist in FL.
We implemented 3 strategies in this repo:

## Implemented Strategies

| Method    | Aggregation Rule | Pros | Cons |
|-----------|----------------|------|------|
| **FedAvg** | Weighted mean of client updates | Simple, efficient | Struggles with non-IID data, client drift |
| **FedMedian** | Element-wise median | Robust to outliers, adversarial clients | Higher computation, slower convergence |
| **FedAvgM** | Weighted mean + momentum | Stabilizes training, faster convergence | More complex, additional hyperparameter |

Each method has its strengths depending on the data distribution and the presence of adversarial clients.
**Notice:** If you have **non-IID** data or adversarial clients, **FedMedian** or **FedAvgM** may work better than standard **FedAvg**.



### 1. FedAvg (Federated Averaging)

FedAvg, proposed by [McMahan et al. (2017)](https://arxiv.org/abs/1602.05629) is the most commonly used algorithm in federated learning.
It works as follows:
- Each client trains a local model using its private dataset.
- The server collects model updates (weights or gradients) from participating clients.
- The server **averages** these updates, weighted by the number of samples per client, to create a new global model.
- The new global model is sent back to the clients for the next round of training.


**Mathematical Formula:**
$
w_{t+1} = \sum_{i=1}^{K} \frac{n_i}{N} w_i^t
$
where:
- $ w_{t+1} $ is the updated global model,
- $ K $ is the number of clients,
- $ n_i $ is the number of samples on client $ i $,
- $ N = \sum_{i=1}^{K} n_i $ is the total number of samples across clients,
- $ w_i^t $ is the local model update from client $ i $ at time $ t $.

**Pros:**
- Simple and effective for IID (independent and identically distributed) data.
- Reduces communication cost compared to frequent gradient updates.

**Cons:**
- Struggles with non-IID data (when clients have different data distributions).
- Can suffer from client drift if local models deviate significantly.


### 2. FedMedian (Federated Median)

This method was proposed by [Yin et at. (2018)](https://arxiv.org/abs/1803.01498).
Fedmedian takes the **element-wise median** of the updates to create the global model, instead of averaging client updates as in FedAvg.
This makes it more robust to malicious or noisy clients.

**Mathematical Formula:**
$
w_{t+1} = \text{median}(\{w_i^t\}_{i=1}^{K})
$
where the median is computed element-wise across all dimensions of the model parameters.

**Pros:**
- More robust to outliers and adversarial clients.
- Can be beneficial in non-IID settings.

**Cons:**
- Computationally more expensive than FedAvg.
- Slower convergence in some cases.

---

### 3. FedAvgM (Federated Averaging with Momentum)

This methods was proposed by [Hsu et al. (2019)](https://arxiv.org/abs/1909.06335).
FedAvgM is an extension of FedAvg that incorporates **server-side momentum** to stabilize training, especially in **non-IID** settings.
The server maintains a momentum term to smooth out fluctuations in FedAvgM.

**Mathematical Formula:**
$
v_{t+1} = \mu v_t + \sum_{i=1}^{K} \frac{n_i}{N} (w_i^t - w_t)
$
$
w_{t+1} = w_t + v_{t+1}
$
where:
- $ v_t $ is the momentum term,
- $ \mu $ is the momentum coefficient (typically 0.9),
- $ w_t $ is the global model at time $ t $,
- $ w_i^t $ is the local update from client $ i $.

**Pros:**
- Improves convergence speed and stability.
- Helps mitigate issues caused by non-IID data.
- More robust to local updates that vary significantly.

**Cons:**
- Introduces an extra hyperparameter (momentum coefficient).
- Slightly higher computation cost at the server.
