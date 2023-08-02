# A ResNet34 to classify objects in the CIFAR100 Dataset

## FINE TUNING OF TENSORBOARD VISUALIZATIONS AND HYPERPARAMETERS IS STILL UNDERWAY
## To be added is a final confusion matrix for a small, isolated part of the test dataset.

## An easily extensible and modifiable code to classify objects in the CIFAR100 Datasets using ResNet34
Some aspects of programming design I recently learned have been applied in the project.
For example: 
- The dataloaders are created by a separate function, exhibiting a _factory pattern_ (a similar thing is done to generate directories for the logs).
- The important metrics (Training Accuracy, Validation Accuracy, Learning Rate) for the project are handled by a `Metric` dataclass.
- The hyperparameter configuration can be done by just editing the file `conf/config.yaml`, the contents of which are made into a dataclass and fed into the `main()` function using the `hydra-core` package (which additionally allows us to automatically structure the results of our experiments according to date and time, saving us an incredible amount of effort in tracking the changes resulting from hyperparameter tuning).

A heavy inspiration for this project is a [simpler project](https://github.com/ArjanCodes/2021-data-science-refactor/tree/main) for classifying handwritten numbers from the **MNIST** dataset. I have heavily modified the code to play properly with _CNN-ResNets_ for _CIFAR-100_ and use **Transfer Learning** and **Learning Rate Schedulers**. I also introduced some basic GPU-memory management since my code expects a **GPU** to run, while the original code did not.

# Running the Experiment
Install packages in an existing environment using `pip`
```shell
pip install -r requirements.txt
```

Download the data
```
python ./ds/download_data.py
```

Run the experiment
```shell
python main.py
```

The hyperparamters can be edited in the `conf/config.yaml` file.
The current batch size of **128** expects **$\geq$ 8GB** of VRAM during the training

# Tensorboard

To launch Tensorboard
```shell
tensorboard --logdir runs
```

The output will be something like:
```shell
TensorBoard 2.6.0 at http://localhost:6006/ (Press CTRL+C to quit)
```

Follow the instructions printed to the terminal to view Tensorboard in a browser.
