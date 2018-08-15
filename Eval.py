""" Testing the network on a single GPU """

from __future__ import absolute_import  # import multi line and Absolute/Relative
from __future__ import division  # change the division operator to output float if dividing two integers
from __future__ import print_function  # use the print function from python 3

import time, os, glob, hashlib

import Model as network
import numpy as np
import tensorflow as tf
import SODTester as SDT
import tensorflow.contrib.slim as slim
import SODLoader as SDL

sdl = SDL.SODLoader(data_root='data/')

_author_ = 'Simi'

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

"""
Treatment groups:
Made 1270 Normal ORIG boxes from 1270 patients. Size: 1270 
Made 1396 Normal FU boxes from 1396 patients. Size: 1396 
Made 508 Treated ORIG boxes from 508 patients. Size: 508 
Made 710 Treated FU boxes from 710 patients. Size: 710  
Followups:
Made 1073 Analysis 2 Normal FU boxes from 1073 patients. Size: 1073 
Made 498 Analysis 2 Treated FU boxes from 498 patients. Size: 498 

CC Only
Made 636 Normal ORIG boxes from 636 patients. Size: 636 
Made 704 Normal FU boxes from 704 patients. Size: 704 
Made 254 Treated ORIG boxes from 254 patients. Size: 254 
Made 352 Treated FU boxes from 352 patients. Size: 352 
CC Followups:
Made 258 Normal ORIG boxes from 258 patients. Size: 258 
Made 119 Treated ORIG boxes from 119 patients. Size: 119 
"""

tf.app.flags.DEFINE_integer('epoch_size', 119, """Test examples: OF: 508""")
tf.app.flags.DEFINE_integer('batch_size', 7, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_classes', 2, """ Number of classes + 1 for background""")
tf.app.flags.DEFINE_string('test_files', 'Treated_FU', """Files for testing have this name""")
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
tf.app.flags.DEFINE_string('train_dir', 'testing/', """Directory to write event logs and save checkpoint files""")
tf.app.flags.DEFINE_string('RunInfo', 'Res_Class_warp3/', """Unique file name for this training run""")
tf.app.flags.DEFINE_integer('GPU', 0, """Which GPU to use""")


# Define a custom training class
def eval():

    # Makes this the default graph where all ops will be added
    with tf.Graph().as_default(), tf.device('cpu:0'):

        # Get a dictionary of our images, id's, and labels here. Use the CPU
        _, valid = network.inputs(skip=True)

        # Define phase of training
        phase_train = tf.placeholder(tf.bool)

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
            avg_softmax, ground_truth, right, total, save_Data = [], [], 0, 0, {}

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
                        preds, labs, patient, group = sess.run([logits, labels, valid['patient'], valid['group']], feed_dict={phase_train: False})

                        # Convert to numpy arrays
                        predictions, label = preds.astype(np.float32), np.squeeze(labs.astype(np.float32))

                        # If first step then create the tracking
                        if i == 0:
                            label_track, logit_track = np.copy(label), np.copy(predictions)
                            pt_track, grp_track = np.copy(patient), np.copy(group)
                        else:
                            label_track, logit_track = np.concatenate((label_track, label)), np.concatenate((logit_track, predictions))
                            pt_track, grp_track = np.concatenate((pt_track, patient)), np.concatenate((grp_track, group))

                # Print errors

                if FLAGS.net_type==0:
                    acc = 100 * (right/total)
                    print ('\nRight this batch: %s, Total: %s, Acc: %0.3f\n' %(right, total, acc))
                else:
                    print (logit_track.shape, label_track.shape)
                    _, label_track, logit_track = sdt.combine_predictions(label_track, logit_track, (pt_track + grp_track), len(logit_track))
                    print(logit_track.shape, label_track.shape)

                    calc_labels = np.squeeze(label_track.astype(np.int8))
                    calc_logit = np.squeeze(np.argmax(logit_track.astype(np.float), axis=1))
                    logit_track = sdt.calc_softmax(logit_track)
                    right, total = 0, 0
                    for z in range (logit_track.shape[0]):
                        if calc_labels[z] == calc_logit[z]: # All positive in this case, save
                            right += 1
                            save_Data[z] = {'ID': z, 'PT': pt_track[z].decode("utf-8"), 'GRP': grp_track[z].decode("utf-8"),
                                            'Class 1': logit_track[z][0], 'Class 2': logit_track[z][1]}
                        total +=1
                    acc = 100 * (right / total)
                    print('\nRight this batch: %s, Total: %s, Acc: %0.3f\n' % (right, total, acc))

                    sdt.save_dic_csv(save_Data, ('%s.csv' %FLAGS.test_files), transpose=True)




def main(argv=None):
    eval()


if __name__ == '__main__':
    tf.app.run()