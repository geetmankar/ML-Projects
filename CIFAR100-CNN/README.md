# An easily extensible and modifiable code to classify objects in the CIFAR100 Datasets using ResNet34

## FINE TUNING OF MODEL HYPERPARAMETERS IS STILL UNDERWAY

Some aspects of programming design I recently learned have been applied in the project.
For example: 
- The dataloaders are created by a separate function, exhibiting a _factory pattern_ (a similar thing is done to generate directories for the logs).
- The important metrics (Training Accuracy, Validation Accuracy, Learning Rate) for the project are handled by a `Metric` dataclass.
- The hyperparameter configuration can be done by just editing the file `conf/config.yaml`, the contents of which are made into a dataclass and fed into the `main()` function using the `hydra-core` package (which additionally allows us to automatically structure the results of our experiments according to date and time, saving us an incredible amount of effort in tracking the changes resulting from hyperparameter tuning).
- The code can be easily extended to use a different CNN model, even for a different image dataset. With moderate amount of modifications, it can easily be changed to train networks for non-image data and models. This is due to the Tracking and Configuration structure being generally model-agnostic.

A heavy inspiration for this project is a [simpler project](https://github.com/ArjanCodes/2021-data-science-refactor/tree/main) for classifying handwritten numbers from the **MNIST** dataset. I have heavily modified the code to play properly with _CNN-ResNets_ for _CIFAR-100_. The modifications are:
- Use of **Transfer Learning** due to a much larger Convolutional network.
- Use of **Learning Rate Schedulers** to control the learning rate across the training process.
- Use of **Gradient-Clipping** for avoiding big jumps over the loss surface.
- Introduced GPU-memory management since my code expects a **GPU** to run, while the original code did not (NOTE: Default Batch-size of 128 corresponds to more than 8GB of VRAM).
- Also changed the DataLoaders to work correctly with GPUs.
- Used `hydra-core` (mentioned above) for better tracking of the changes in the results.
- Added a Test set (apart from the validation set), and added a confusion matrix to represent the results of the same.

---

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
