""" Testing the network on a single GPU """

from __future__ import absolute_import  # import multi line and Absolute/Relative
from __future__ import division  # change the division operator to output float if dividing two integers
from __future__ import print_function  # use the print function from python 3

import time, os, glob

import Model as network
import numpy as np
import tensorflow as tf
import SODTester as SDT
import tensorflow.contrib.slim as slim
import SODLoader as SDL
import SOD_Display as SDD

sdl = SDL.SODLoader(data_root='data/')
sdd = SDD.SOD_Display()

_author_ = 'Simi'

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the immutable variables
tf.app.flags.DEFINE_integer('num_classes', 2, """ Number of classes + 1 for background""")
tf.app.flags.DEFINE_string('test_files', '2', """Files for testing have this name""")
tf.app.flags.DEFINE_integer('box_dims', 64, """dimensions of the input pictures""")
tf.app.flags.DEFINE_integer('network_dims', 64, """the dimensions fed into the network""")
tf.app.flags.DEFINE_integer('epoch_size', 63, """How many images were loaded""")
tf.app.flags.DEFINE_integer('batch_size', 21, """Number of images to process in a batch.""")

# Regularizers
tf.app.flags.DEFINE_float('dropout_factor', 1.0, """ Keep probability""")
tf.app.flags.DEFINE_float('l2_gamma', 1e-4, """ The gamma value for regularization loss""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.999, """ The decay rate for the moving average tracker""")

# Directory control
tf.app.flags.DEFINE_string('train_dir', 'training/', """Directory to write event logs and save checkpoint files""")
tf.app.flags.DEFINE_string('RunInfo', 'Run1/', """Unique file name for this training run""")
tf.app.flags.DEFINE_integer('GPU', 0, """Which GPU to use""")
tf.app.flags.DEFINE_integer('sleep_time', 0, """How long to wait before starting processing""")


# Define a custom training class
def eval():

    # Makes this the default graph where all ops will be added
    with tf.Graph().as_default(), tf.device('cpu:0'):

        # Get a dictionary of our images, id's, and labels here. Use the CPU
        _, valid = network.inputs(skip=True)

        # Define phase of training
        phase_train = tf.placeholder(tf.bool)

        # Build a graph that computes the prediction from the inference model (Forward pass)
        logits, _ = network.forward_pass(valid['data'], phase_train=phase_train)
        labels = valid['label']

        # Initialize variables operation
        var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Restore moving average of the variables
        var_ema = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay)

        # Define variables to restore
        var_restore = var_ema.variables_to_restore()

        # Initialize the saver
        saver = tf.train.Saver(var_restore, max_to_keep=3)

        # Trackers for best performers
        best_MAE, best_epoch = 0, 0

        while True:

            # Allow memory placement growth
            config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:

                # Retreive the checkpoint
                ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir + FLAGS.RunInfo)

                # Initialize the variables
                sess.run(var_init)

                if ckpt and ckpt.model_checkpoint_path:

                    # Restore the learned variables
                    restorer = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')

                    # Restore the graph
                    restorer.restore(sess, ckpt.model_checkpoint_path)

                    # Extract the epoch
                    Epoch = ckpt.model_checkpoint_path.split('/')[-1].split('_')[-1]

                # Set the max step count
                max_steps = int(FLAGS.epoch_size / FLAGS.batch_size)

                # Define tester class instance
                sdt = SDT.SODTester(True, False)

                # Use slim to handle queues:
                with slim.queues.QueueRunners(sess):

                    for i in range(max_steps):

                        # Also retreive the predictions and labels
                        preds, labs, unq, all = sess.run([logits, labels, valid['mrn'], valid], feed_dict={phase_train: False})

                        # Convert to numpy arrays
                        predictions, label, unique = preds.astype(np.float32), np.squeeze(labs.astype(np.float32)), np.squeeze(unq)

                        # If first step then create the tracking
                        if i == 0:
                            label_track = np.copy(label)
                            logit_track = np.copy(predictions)
                            unique_track = np.copy(unique)
                        else:
                            label_track = np.concatenate((label_track, label))
                            logit_track = np.concatenate((logit_track, predictions))
                            unique_track = np.concatenate((unique_track, unique))

                    # Print errors
                    print (logit_track.shape, label_track.shape, end='')
                    _, label_track, logit_track = sdt.combine_predictions(label_track, logit_track, unique_track, FLAGS.batch_size)
                    print(logit_track.shape, label_track.shape)
                    sdt.calculate_metrics(np.asarray(logit_track), np.asarray(label_track), 1, max_steps)
                    sdt.retreive_metrics_classification(Epoch)

                    # Lets save runs that are best
                    if sdt.AUC*100 >= best_MAE:

                        # Save the checkpoint
                        print(" ---------------- SAVING THIS ONE %s", ckpt.model_checkpoint_path)

                        # Define the filename
                        file = ('Epoch_%s_AUC_%0.3f' % (Epoch, sdt.AUC))

                        # Define the checkpoint file:
                        checkpoint_file = os.path.join('testing/' + FLAGS.RunInfo, file)

                        # Save the checkpoint
                        saver.save(sess, checkpoint_file)

                        # Save a new best MAE
                        best_MAE = sdt.AUC*100
                        best_epoch = Epoch

            # Otherwise check folder for changes
            filecheck = glob.glob(FLAGS.train_dir + FLAGS.RunInfo + '*')
            newfilec = filecheck

            # Sleep if no changes
            while filecheck == newfilec:

                # Sleep an amount of time proportional to the epoch size
                #time.sleep(60*120)

                # Recheck the folder for changes
                newfilec = glob.glob(FLAGS.train_dir + FLAGS.RunInfo + '*')



def main(argv=None):  # pylint: disable=unused-argument
    time.sleep(FLAGS.sleep_time)
    if tf.gfile.Exists('testing/' + FLAGS.RunInfo):
        tf.gfile.DeleteRecursively('testing/' + FLAGS.RunInfo)
    tf.gfile.MakeDirs('testing/' + FLAGS.RunInfo)
    eval()


if __name__ == '__main__':
    tf.app.run()