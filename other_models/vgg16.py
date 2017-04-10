# -*- coding: utf-8 -*-
"""VGG16 model for Keras."""
import keras

model = keras.applications.VGG16()
model.save("vgg16.h5")
