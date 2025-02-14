This repository contains experiments with federated learning frameworks.


# Installation

## For `py-flwr`

```
cd fedl
pyenv virtualenv 3.11 fedl
pyenv activate fedl
pip3 install -r py-flwr/requirements.txt
```


The following command can be used to verify if Flower.ai framework was successfully installed.
If everything worked, it should print the version of Flower to the command line:

```
python -c "import flwr;print(flwr.__version__)"
```
