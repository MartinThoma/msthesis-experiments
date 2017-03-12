#!/usr/bin/env python

"""Train a model with ADAM."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from datetime import datetime
import time
# import math
# import numpy as np
import os

import tensorflow as tf

MODEL_NAME = 'test'


def loss_function(logits, labels):
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
    single_errors = labels * tf.log(logits + 10**(-7))
    cross_entropy_mean = tf.reduce_mean(-tf.reduce_sum(single_errors,
                                                       reduction_indices=[1]),
                                        name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the
    # weight decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def eval_network(sess, summary_writer, dataset, correct_prediction, epoch,
                 mode, x, y_):
    """Evaluate the network."""
    correct_sum = 0
    total_test = 0
    batch_size = 1000
    for i in range(int(dataset.labels.shape[0] / batch_size)):
        feed_dict = {x: dataset.images[i * batch_size:(i + 1) * batch_size],
                     y_: dataset.labels[i * batch_size:(i + 1) * batch_size]}
        test_correct = correct_prediction.eval(feed_dict=feed_dict)
        correct_sum += sum(test_correct)
        total_test += len(test_correct)
    return float(correct_sum) / total_test


def log_score(sess, summary_writer, filename, data, scoring, epoch, x, y_):
    """Write the score to in CSV format to a file."""
    with open(filename, "a") as myfile:
        train = eval_network(sess, summary_writer, data.train, scoring, epoch,
                             "train", x, y_)
        test = eval_network(sess, summary_writer, data.test, scoring, epoch,
                            "test", x, y_)
        myfile.write("%i;%0.6f;%0.6f\n" % (epoch, train, test))


def get_nonexisting_path(model_checkpoint_path):
    """Get a path which no other file uses."""
    if not os.path.isfile(model_checkpoint_path):
        return model_checkpoint_path
    else:
        folder = os.path.dirname(model_checkpoint_path)
        filename = os.path.basename(model_checkpoint_path)
        filename, ext = os.path.splitext(filename)
        i = 1
        gen_filename = os.path.join(folder, "%s-%i%s" % (filename, i, ext))
        while os.path.isfile(gen_filename):
            i += 1
            gen_filename = os.path.join(folder, "%s-%i%s" % (filename, i, ext))
        return gen_filename


def train(data,
          model,
          optimizer,
          train_dir='.',
          log_device_placement=True,
          config={}):
    """Train for a number of steps."""
    train_params = config['train']
    with tf.Session() as sess:
        global_step = tf.contrib.framework.get_or_create_global_step()

        summary_writer = tf.summary.FileWriter('summary_dir', sess.graph)
        val = os.path.join(train_dir,
                           'validation-curve-accuracy-%s.csv' % MODEL_NAME)
        validation_curve_path = get_nonexisting_path(val)

        # Get images and labels
        data.prepare(data.DATA_DIR)
        # data.visualize(42)
        dataset = data.read_data_sets()
        config['dataset']['meta'] = data.meta

        # Build a Graph that computes the logits predictions from the
        # inference model.
        x = tf.placeholder(tf.float32, shape=[None,
                                              (data.meta['image_width'] *
                                               data.meta['image_height'] *
                                               data.meta['image_depth'])])
        y_ = tf.placeholder(tf.float32, shape=[None, data.meta['n_classes']])
        logits = model.inference(x, data.meta)

        # Calculate loss.
        loss = loss_function(logits, y_)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_step = optimizer.train(loss, global_step, config)

        sess.run(tf.global_variables_initializer())

        t0 = time.time()
        examples_per_epoch = config['dataset']['meta']['examples_per_epoch']
        num_batches_per_epoch = int(examples_per_epoch /
                                    train_params['batch_size'])
        for i in range(int(train_params['epochs']) * num_batches_per_epoch):
            batch = dataset.train.next_batch(train_params['batch_size'])
            if i % num_batches_per_epoch == 0:
                log_score(sess, summary_writer,
                          validation_curve_path,
                          dataset, correct_prediction, i, x, y_)
            train_step.run(feed_dict={x: batch[0],
                                      y_: batch[1]
                                      })
        t1 = time.time()
        print("Time: %0.4fs" % (t1 - t0))


def main(data, model, optimizer, experiment_file, config):
    """Orchestrate."""
    train_dir = experiment_file[:-5]
    print("train_dir: %s" % train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if tf.gfile.Exists(train_dir):
        tf.gfile.DeleteRecursively(train_dir)
    tf.gfile.MakeDirs(train_dir)
    train(data, model, optimizer, train_dir=train_dir,
          config=config)


if __name__ == '__main__':
    main()
