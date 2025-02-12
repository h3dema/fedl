
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
