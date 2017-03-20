# msthesis-experiments
Experiments for my masters thesis

Every experiment is a YAML file

```
dataset_path: datasets/cifar10.py
model_path: models/mlp.py
optimizer_path: optimizers/adamdef.py
train_path: train/train.py
```

* **dataset** contains
    * `inputs(eval_data, data_dir, batch_size)` and
    * `distorted_inputs(ata_dir=DATA_DIR, batch_size=128)` and
    * a dict `meta` which contains `n_classes`
* **model** contains `inference(images, dataset_meta)`
* **optimizer** contains `train(total_loss, global_step)`
* **train** contains `train(data, model, optimizer)`

## Plan

* GTSBM
* MNIST / SVHN
* HASY
* How do humans group classes?
* Which groups form -> Image in thesis!
* Why do I use such an approach:
    - Faster evaluation of ideas due to faster training due to less data
    - Easier to estimate the benefit of more data (and which kind of data)
    - Lower labeling cost


* CIFAR: https://cambridgespark.com/content/tutorials/convolutional-neural-networks-with-keras/index.html