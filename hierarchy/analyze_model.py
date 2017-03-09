#!/usr/bin/env python


"""Analyze a cifar100 keras model."""

from keras.models import load_model
from keras.datasets import cifar100
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import json
import io
import os
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

n_classes = 100

# # Load label names
if os.path.isfile("meta"):
    with open("meta", "rb") as f:
        data = pickle.load(f)
    labels = data['fine_label_names']

    # Store label names as json
    with io.open('cifar-100-labels.json', 'w', encoding='utf8') as outfile:
        str_ = json.dumps(labels,
                          indent=4, sort_keys=True,
                          separators=(',', ':'), ensure_ascii=False)
        outfile.write(to_unicode(str_))
with open('cifar-100-labels.json') as data_file:
    labels = json.load(data_file)

# Load model
model = load_model('cifar100.h5')

# Load validation data
(X, y), (X_test, y_test) = cifar100.load_data()

import matplotlib.pyplot as plt
data = y.flatten()
plt.hist(data, bins=np.arange(data.min(), data.max() + 2))
plt.savefig("data_dist.png", dpi=300)

X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=0.20,
                                                  random_state=42)

# Calculate confusion matrix
y_val_i = y_val.flatten()
y_val_pred = model.predict(X_val)
y_val_pred_i = y_val_pred.argmax(1)
cm = np.zeros((n_classes, n_classes), dtype=np.int)
for i, j in zip(y_val_i, y_val_pred_i):
    cm[i][j] += 1

# Serialize confusion matrix
with io.open('cm.json', 'w', encoding='utf8') as outfile:
    str_ = json.dumps(cm.tolist(),
                      indent=4, sort_keys=True,
                      separators=(',', ':'), ensure_ascii=False)
    outfile.write(to_unicode(str_))
