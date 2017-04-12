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

## Requirements

* matplotlib
* Tensorflow 1.0
* Keras 2.0 (adjusted [preprocessing/image.py](https://github.com/fchollet/keras/pull/6003), see misc directory)
* seaborn

## Run timining experiments

Measuring inference time needs about 2 minutes:

```
$ ./inference_timing.py -f experiments/cifar10_baseline.yaml
```

Measuring training time takes about 70 minutes:

```
$ ./run_training.py -f experiments/cifar10_baseline.yaml
```