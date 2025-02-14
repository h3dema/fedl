This repository contains experiments with federated learning frameworks.


# Installation

## For `pytorch-example`

```
cd fedl/py-fedl
pyenv virtualenv 3.11 fedl
pyenv activate fedl
pip3 install -r requirements.txt
```


The following command can be used to verify if Flower.ai framework was successfully installed.
If everything worked, it should print the version of Flower to the command line:

```
python -c "import flwr;print(flwr.__version__)"
```
