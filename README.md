# msthesis-experiments
Experiments for the masters thesis of Martin Thoma

Every experiment is a YAML file. See the experiments folder for some examples.


## Requirements

* matplotlib
* Tensorflow 1.0
* Keras 2.0 (adjusted [preprocessing/image.py](https://github.com/fchollet/keras/pull/6003), see misc directory)
* seaborn

If you get `TypeError: __init__() got an unexpected keyword argument
'hsv_augmentation'` you didn't adjust the `image.py`. Just copy the one in
the misc folder to `python -c "import keras.preprocessing.image as k;print(k.__file__)"`


## Scripts

* `./run_training.py -f experiments/cifar100_baseline.yaml`: Train a model. Downloads everything by its own
* `./analyze_training.py -d artifacts/cifar100_baseline`: Show some training statistics
* `./inference_timing.py -f experiments/cifar100_baseline.yaml`: Run inference on a given trained model and measure the time
* `./eval_ensemble.py -f ensemble/cifar100_baseline.yaml`: Evaluate an ensemble
* `./visualize.py --cm artifacts/cifar100_root/cm-test.json`: Confusion matrix optimization
* `./create_cm.py --indices cm.indices.pickle -f experiments/cifar100_root-g5.yaml`

## Run timining experiments

Measuring inference time needs about 2 minutes:

```
$ ./inference_timing.py -f experiments/cifar10_baseline.yaml
```

Measuring training time takes about 70 minutes:

```
$ ./run_training.py -f experiments/cifar10_baseline.yaml
```