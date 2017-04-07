#!/usr/bin/env python

import numpy as np


def make_mosaic(imgs, nrows, ncols, border=1, mode='tf'):
    """
    Make mosaik.

    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    if mode == 'tf':
        imgs = imgs.transpose(2, 0, 1)
    imshape = imgs.shape[1:]

    if len(imgs.shape) == 4:
        if imgs.shape[-1] >= 3:
            channels = 3
        else:
            channels = imgs.shape[-1]
    else:
        channels = 1

    mosaic = np.zeros((nrows * imshape[0] + (nrows - 1) * border,
                       ncols * imshape[1] + (ncols - 1) * border,
                       channels),
                      dtype=np.float32)

    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in range(nrows * ncols):
        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1], 0] = imgs[i]
    return mosaic.squeeze()
