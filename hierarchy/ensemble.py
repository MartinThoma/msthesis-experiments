#!/usr/bin/env python


"""Build an ensemble of CIFAR 100 Keras models."""

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.models import load_model
from keras.datasets import cifar100
from sklearn.model_selection import train_test_split
import numpy as np
from analyze_model import plot_cm
import glob
from keras.utils import np_utils
from natsort import natsorted


def calculate_cm(y_true, y_pred):
    """Calculate confusion matrix."""
    y_true_i = y_true.flatten()
    y_pred_i = y_pred.argmax(1)
    cm = np.zeros((n_classes, n_classes), dtype=np.int)
    for i, j in zip(y_true_i, y_pred_i):
        cm[i][j] += 1
    return cm

n_classes = 100

# Load models
# models = ['cifar100-%i.h5' % i for i in range(1, 5)]
model_names = natsorted(glob.glob("cifar100-*.h5"))
print("Ensemble of {} models ({})".format(len(model_names), model_names))
models = [load_model(model_path) for model_path in model_names]

# Load data
(X, y), (X_test, y_test) = cifar100.load_data()

X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=0.20,
                                                  random_state=42)
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_val /= 255
X_test /= 255


# Calculate confusion matrix
y_val_i = y_val.flatten()
y_preds = [model.predict(X_train) for model in models]

for model_index, y_val_pred in enumerate(y_preds):
    cm = calculate_cm(y_train, y_val_pred)
    acc = sum([cm[i][i] for i in range(100)]) / float(cm.sum())
    print("Cl #{} ({}): accuracy: {:0.2f}".format(model_index + 1,
                                                  model_names[model_index],
                                                  acc * 100))

y_val_pred = sum(y_preds) / len(model_names)
cm = calculate_cm(y_train, y_val_pred)
acc = sum([cm[i][i] for i in range(100)]) / float(cm.sum())
print("Ensemble Accuracy: %0.2f" % (acc * 100))

Y_train = np_utils.to_categorical(y_train, n_classes)
smoothed_lables = (y_val_pred + Y_train) / 2
np.save("smoothed_lables", smoothed_lables)

# Create plot
plot_cm(cm)
