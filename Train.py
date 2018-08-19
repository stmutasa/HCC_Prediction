""" Training the network on a single GPU """

from __future__ import absolute_import      # import multi line and Absolute/Relative
from __future__ import division             # change the division operator to output float if dividing two integers
from __future__ import print_function       # use the print function from python 3

import os
import time                                 # to retreive current time
import numpy as np
import tensorflow.contrib.slim as slim

import Model as network
import tensorflow as tf

_author_ = 'Simi'

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the immutable variables
tf.app.flags.DEFINE_integer('num_classes', 2, """ Number of classes""")
tf.app.flags.DEFINE_string('test_files', 'Final', """Files for testing have this name""")
tf.app.flags.DEFINE_integer('box_dims', 64, """dimensions of the input pictures""")
tf.app.flags.DEFINE_integer('network_dims', 64, """the dimensions fed into the network""")

#
tf.app.flags.DEFINE_integer('epoch_size', 252, """How many images were loaded""")
tf.app.flags.DEFINE_integer('num_epochs', 900, """Number of epochs to run""")
tf.app.flags.DEFINE_integer('print_interval', 5, """How often to print a summary to console during training""")
tf.app.flags.DEFINE_integer('checkpoint_interval', 50, """How many Epochs to wait before saving a checkpoint""")
tf.app.flags.DEFINE_integer('batch_size', 70, """Number of images to process in a batch.""")

# Regularizers
tf.app.flags.DEFINE_float('dropout_factor', 0.75, """ Keep probability""")
tf.app.flags.DEFINE_float('l2_gamma', 1e-4, """ The gamma value for regularization loss""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.999, """ The decay rate for the moving average tracker""")

# Hyperparameters to control the learning rate
tf.app.flags.DEFINE_float('learning_rate', 1e-3, """Initial learning rate""")
tf.app.flags.DEFINE_float('beta1', 0.9, """ The beta 1 value for the adam optimizer""")
tf.app.flags.DEFINE_float('beta2', 0.999, """ The beta 1 value for the adam optimizer""")

# Directory control
tf.app.flags.DEFINE_string('train_dir', 'training/', """Directory to write event logs and save checkpoint files""")
tf.app.flags.DEFINE_string('RunInfo', 'Run1/', """Unique file name for this training run""")
tf.app.flags.DEFINE_integer('GPU', 0, """Which GPU to use""")
tf.app.flags.DEFINE_integer('sleep_time', 0, """How long to wait before starting processing""")


def train():

    # Makes this the default graph where all ops will be added
    with tf.Graph().as_default(), tf.device('/gpu:' + str(FLAGS.GPU)):

        # Get a dictionary of our images, id's, and labels here. Use the CPU
        with tf.device('/cpu:0'): images, _ = network.inputs(skip=True)

        # Define phase of training
        phase_train = tf.placeholder(tf.bool)

        # Run the network depending on type
        logits, l2loss = network.forward_pass(images['data'], phase_train=phase_train)
        labels = images['label']
        SCE_loss = network.sdn.SCE_loss(logits, labels, FLAGS.num_classes)

        # Add in L2 Regularization
        loss = tf.add(SCE_loss, l2loss, name='loss')

        # Update the moving average batch norm ops
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Retreive the training operation with the applied gradients
        with tf.control_dependencies(extra_update_ops): train_op = network.backward_pass(loss)

        # Merge the summaries
        all_summaries = tf.summary.merge_all()

        # Initialize variables operation
        var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Restore moving average of the variables
        var_ema = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay)

        # Define variables to restore
        var_restore = var_ema.variables_to_restore()

        # Initialize the saver
        saver = tf.train.Saver(var_restore, max_to_keep=10)

        # Set the intervals
        max_steps = FLAGS.num_epochs * FLAGS.epoch_size // FLAGS.batch_size
        print_interval = FLAGS.print_interval * FLAGS.epoch_size // FLAGS.batch_size
        checkpoint_interval = FLAGS.checkpoint_interval * FLAGS.epoch_size // FLAGS.batch_size
        print('Max Steps: %s, Print Interval: %s, Checkpoint: %s' % (max_steps, print_interval, checkpoint_interval))
        print('\nGPU: %s, File:%s' % (FLAGS.GPU, FLAGS.RunInfo[:-1]))

        # Allow memory placement growth
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:

            # Init global variables
            sess.run(tf.global_variables_initializer())

            # Initialize the handle to the summary writer in our training directory
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir + FLAGS.RunInfo, sess.graph)

            # Initialize timers
            timer = 0

            # Use slim to handle queues:
            with slim.queues.QueueRunners(sess):

                for i in range(max_steps):

                    # Run an iteration. Ramp down Gen:Dis steps
                    start = time.time()
                    sess.run(train_op, feed_dict={phase_train: True})
                    timer += (time.time() - start)

                    # What epoch
                    Epoch = int((i * FLAGS.batch_size) / FLAGS.epoch_size)

                    # print interval
                    if i % print_interval == 0:

                        # Get timing stats
                        elapsed = timer / print_interval
                        timer = 0

                        # Load some metrics
                        loss1, loss2, tot = sess.run([SCE_loss, l2loss, loss], feed_dict={phase_train: True})

                        # use numpy to print only the first sig fig
                        np.set_printoptions(precision=2)

                        # Print the data
                        print('-'*70, '\nEpoch %d, L2 Loss: = %.3f (%.1f eg/s;), Total Loss: %.3f SCE: %.4f'
                              % (Epoch, loss2, FLAGS.batch_size / elapsed, tot, loss1))

                        # Run a session to retrieve our summaries
                        summary = sess.run(all_summaries, feed_dict={phase_train: True})

                        # Add the summaries to the protobuf for Tensorboard
                        summary_writer.add_summary(summary, i)


                    if i % checkpoint_interval == 0:

                        print('-' * 70, '\nSaving... GPU: %s, File:%s' % (FLAGS.GPU, FLAGS.RunInfo[:-1]))

                        # Define the filename
                        file = ('Epoch_%s' % Epoch)

                        # Define the checkpoint file:
                        checkpoint_file = os.path.join(FLAGS.train_dir + FLAGS.RunInfo, file)

                        # Save the checkpoint
                        saver.save(sess, checkpoint_file)

                        time.sleep(0)


def main(argv=None):  # pylint: disable=unused-argument
    time.sleep(FLAGS.sleep_time)
    if tf.gfile.Exists(FLAGS.train_dir + FLAGS.RunInfo):
        tf.gfile.DeleteRecursively(FLAGS.train_dir + FLAGS.RunInfo)
    tf.gfile.MakeDirs(FLAGS.train_dir + FLAGS.RunInfo)
    train()

if __name__ == '__main__':
    tf.app.run()