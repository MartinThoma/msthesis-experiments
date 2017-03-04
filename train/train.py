#!/usr/bin/env python

"""Train a model with ADAM."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import sys
sys.path.insert(1, '/home/moose/GitHub/msthesis-experiments/models')
import mlp as model
sys.path.insert(1, '/home/moose/GitHub/msthesis-experiments/datasets')
import cifar10_input as data


def loss(logits, labels):
    """
    Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Parameters
    ----------
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]

    Returns
    -------
    Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the
    # weight decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train(batch_size=16, train_dir='.', max_steps=1000,
          log_device_placement=True):
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Get images and labels for CIFAR-10.
        images, labels = data.distorted_inputs()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = model.inference(images)

        # Calculate loss.
        loss = cifar10.loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = cifar10.train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1

            def before_run(self, run_context):
                self._step += 1
                self._start_time = time.time()
                return tf.train.SessionRunArgs(loss)    # Asks for loss value.

            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time
                loss_value = run_values.results
                if self._step % 10 == 0:
                    num_examples_per_step = batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.2f '
                                  '(%.1f examples/sec; %.3f sec/batch)')
                    print(format_str % (datetime.now(), self._step,
                          loss_value,
                          examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=train_dir,
                hooks=[tf.train.StopAtStepHook(last_step=max_steps),
                       tf.train.NanTensorHook(loss),
                       _LoggerHook()],
                config=tf.ConfigProto(log_device_placement=log_device_placement)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def main(train_dir='traindir'):
    data.maybe_download_and_extract()
    if tf.gfile.Exists(train_dir):
        tf.gfile.DeleteRecursively(train_dir)
    tf.gfile.MakeDirs(train_dir)
    train(train_dir=train_dir)


if __name__ == '__main__':
    main()
