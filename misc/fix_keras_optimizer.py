#!/usr/bin/env python

"""Make keras 1.x models usable in keras 2.x."""

import glob
import h5py

model_files = sorted(glob.glob('*.h5'))
for model_file in model_files:
    print("Update '{}'".format(model_file))
    with h5py.File(model_file, 'a') as f:
        if 'optimizer_weights' in f.keys():
            del f['optimizer_weights']
