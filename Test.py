""" Testing the network on a single GPU """

from __future__ import absolute_import  # import multi line and Absolute/Relative
from __future__ import division  # change the division operator to output float if dividing two integers
from __future__ import print_function  # use the print function from python 3

import time, os, glob

import HCC_Matrix as network
import numpy as np
import tensorflow as tf
import SODTester as SDT
import tensorflow.contrib.slim as slim
import SODLoader as SDL

sdl = SDL.SODLoader(data_root='data/')

_author_ = 'Simi'

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('epoch_size', 585, """Test examples: OF: 508""")
tf.app.flags.DEFINE_integer('batch_size', 5, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_classes', 2, """ Number of classes + 1 for background""")
tf.app.flags.DEFINE_string('test_files', 'Test', """Files for testing have this name""")
tf.app.flags.DEFINE_integer('box_dims', 512, """dimensions of the input pictures""")
tf.app.flags.DEFINE_integer('network_dims', 256, """the dimensions fed into the network""")
tf.app.flags.DEFINE_integer('net_type', 1, """ 0=Segmentation, 1=classification """)

# Hyperparameters:
tf.app.flags.DEFINE_float('dropout_factor', 1.0, """ p value for the dropout layer""")
tf.app.flags.DEFINE_float('l2_gamma', 1e-4, """ The gamma value for regularization loss""")
tf.app.flags.DEFINE_float('loss_factor', 1.0, """Penalty for missing a class is this times more severe""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.999, """ The decay rate for the moving average tracker""")
tf.app.flags.DEFINE_float('threshold', 0.464, """Softmax threshold for declaring cancer""")

# Directory control
tf.app.flags.DEFINE_string('train_dir', 'training/', """Directory to write event logs and save checkpoint files""")
tf.app.flags.DEFINE_string('RunInfo', 'Retrain/', """Unique file name for this training run""")
tf.app.flags.DEFINE_integer('GPU', 0, """Which GPU to use""")


# Define a custom training class
def eval():

    # Makes this the default graph where all ops will be added
    with tf.Graph().as_default(), tf.device('cpu:0'):

        # Get a dictionary of our images, id's, and labels here. Use the CPU
        _, valid = network.inputs(skip=True)

        # Define phase of training
        phase_train = tf.placeholder(tf.bool)

        # Build a graph that computes the prediction from the inference model (Forward pass)


        # Run the network depending on type
        if FLAGS.net_type == 0:
            logits, _ = network.forward_pass(valid['data'], phase_train=phase_train)
            softmax = tf.nn.softmax(logits)
            labels = valid['label_data']
        else:
            logits, _ = network.forward_pass_class(valid['data'], phase_train=phase_train)
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
                if FLAGS.net_type ==0: sdt = SDT.SODTester(False, False)
                else: sdt = SDT.SODTester(True, False)
                avg_softmax, ground_truth, right, total = [], [], 0, 0

                # Use slim to handle queues:
                with slim.queues.QueueRunners(sess):

                    for i in range(max_steps):

                        if FLAGS.net_type ==0: # Segmentation network

                            # Load some metrics for testing
                            lbl1, logtz, imgz, serz, smx = sess.run([labels, logits, valid['data'], valid['patient'], softmax], feed_dict={phase_train: False})
                            label_normalize, smx = np.copy(lbl1), np.squeeze(logtz)

                            # Calculate average softmax
                            for z in range (FLAGS.batch_size):

                                # Append the label to the tracker as one number
                                ground_truth.append(np.amax(label_normalize[z]))

                                # Make mask by setting label background to 0 and breast to 1
                                label_normalize[z][label_normalize[z] >0] = 1

                                # Apply mask to logits. And Make background (class 0) predictions 0
                                smx[z] *= label_normalize[z]
                                smx[z, :, :, 0] *= 0

                                # Generate softmax scores from the two cancer classes
                                softmaxed_output = sdt.calc_softmax(np.reshape(smx[z, :, :, 1:], (-1, (FLAGS.num_classes-1))))

                                # # TODO: Display
                                # display_logits = np.copy(smx[z, :, :, 2])
                                # display_softmax = sdt.calc_softmax(np.copy(smx[z, :, :, 1:]))
                                # sdl.display_single_image(display_logits, False, cmap='jet')
                                # sdl.display_single_image(display_softmax[:, :, 1], cmap='jet')

                                # Make a row of softmax predictions by taking the average prediction for each class. Then add to tracker
                                avg_smx = np.average(softmaxed_output, axis=0)
                                avg_softmax.append(avg_smx)

                                # Increment counters
                                if ground_truth[z] == 2 and avg_smx[1] > FLAGS.threshold: right +=1
                                elif ground_truth[z] == 1 and avg_smx[1] < FLAGS.threshold: right += 1
                                total += 1

                                # Print summary every 10 examples
                                if z%10 == 0: print ('Label: %s, Softmaxes: %s' %(ground_truth[z], avg_smx))

                        else: # Classification network

                            # Also retreive the predictions and labels
                            preds, labs = sess.run([logits, labels], feed_dict={phase_train: False})

                            # Convert to numpy arrays
                            predictions, label = preds.astype(np.float32), np.squeeze(labs.astype(np.float32))

                            # If first step then create the tracking
                            if i == 0:
                                label_track = np.copy(label)
                                logit_track = np.copy(predictions)
                            else:
                                label_track = np.concatenate((label_track, label))
                                logit_track = np.concatenate((logit_track, predictions))

                    # Print errors

                    if FLAGS.net_type==0:
                        acc = 100 * (right/total)
                        print ('\nRight this batch: %s, Total: %s, Acc: %0.3f\n' %(right, total, acc))
                    else:
                        print (logit_track.shape, label_track.shape)
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
    time.sleep(0)
    if tf.gfile.Exists('testing/' + FLAGS.RunInfo):
        tf.gfile.DeleteRecursively('testing/' + FLAGS.RunInfo)
    tf.gfile.MakeDirs('testing/' + FLAGS.RunInfo)
    eval()


if __name__ == '__main__':
    tf.app.run()