#!/usr/bin/env python


"""Build an ensemble of CIFAR 100 Keras models."""

from keras.models import load_model
from keras.datasets import cifar100
from sklearn.model_selection import train_test_split
import numpy as np
from analyze_model import plot_cm
import glob

n_classes = 100

# Load models
# models = ['cifar100-%i.h5' % i for i in range(1, 5)]
models = sorted(glob.glob("cifar100-*.h5"))
print("Ensemble of {} models ({})".format(len(models), models))
models = [load_model(model_path) for model_path in models]

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
y_val_pred = sum([model.predict(X_val) for model in models])
y_val_pred_i = y_val_pred.argmax(1)
cm = np.zeros((n_classes, n_classes), dtype=np.int)
for i, j in zip(y_val_i, y_val_pred_i):
    cm[i][j] += 1

acc = sum([cm[i][i] for i in range(100)]) / float(cm.sum())
print("Accuracy: %0.4f" % acc)

# Create plot
plot_cm(cm)
