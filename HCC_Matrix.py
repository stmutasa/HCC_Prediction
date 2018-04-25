# Defines and builds our network
#    Computes input images and labels using inputs() or distorted inputs ()
#    Computes inference on the models (forward pass) using inference()
#    Computes the total loss using loss()
#    Performs the backprop using train()

from __future__ import absolute_import  # import multi line and Absolute/Relative
from __future__ import division  # change the division operator to output float if dividing two integers
from __future__ import print_function  # use the print function from python 3

_author_ = 'simi'

import tensorflow as tf
import Input
import SODNetwork as SDN

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the immutable variables
tf.app.flags.DEFINE_string('data_dir', 'data/', """Path to the data directory.""")

# Retreive helper function object
sdn = SDN.SODMatrix()

def forward_pass(images, phase_train):

    """
    This function builds the network architecture and performs the forward pass
    Two main architectures depending on where to insert the inception or residual layer
    :param images: Images to analyze
    :param phase_train1: bool, whether this is the training phase or testing phase
    :return: logits: the predicted age from the network
    :return: l2: the value of the l2 loss
    """

    K = 16

    # First layer is conv
    print('Input Images: ', images)
    images = tf.expand_dims(images, -1)

    # # Residual blocks
    # conv = sdn.convolution('Conv1', images, 3, K, 1, phase_train=phase_train)
    # conv = sdn.residual_layer('Residual1a', conv, 3, K * 2, 2, phase_train=phase_train) # 32
    # conv = sdn.residual_layer('Residual1b', conv, 3, K * 2, 1, phase_train=phase_train)
    # conv = sdn.residual_layer('Residual2a', conv, 3, K * 4, 2, phase_train=phase_train) # 16
    # conv = sdn.residual_layer('Residual2b', conv, 3, K * 4, 1, phase_train=phase_train)
    # conv = sdn.residual_layer('Residual3a', conv, 3, K * 8, 2, phase_train=phase_train)  # 8
    # conv = sdn.residual_layer('Residual3b', conv, 3, K * 8, 1, phase_train=phase_train)
    # conv = sdn.inception_layer('Inception', conv, K * 16, S=2, phase_train=phase_train) # 4
    # conv = sdn.residual_layer('Residual4a', conv, 3, K * 16, 1, phase_train=phase_train)
    # conv = sdn.residual_layer('Residual4b', conv, 3, K * 16, 1, phase_train=phase_train)
    # conv = sdn.residual_layer('Residual4c', conv, 3, K * 16, 1, phase_train=phase_train)

    # Define densenet class
    dense = SDN.DenseNet(nb_blocks=6, filters=6, sess=None, phase_train=phase_train, summary=False)
    conv = sdn.convolution('Conv1', images, 3, 16, 1, phase_train=phase_train)
    conv = dense.dense_block(conv, nb_layers=4, layer_name='Dense64', downsample=True)
    conv = dense.dense_block(conv, nb_layers=8, layer_name='Dense32', downsample=True)
    conv = dense.dense_block(conv, nb_layers=16, layer_name='Dense16', downsample=True)
    conv = dense.dense_block(conv, nb_layers=24, layer_name='Dense8', downsample=True)
    conv = dense.dense_block(conv, nb_layers=48, layer_name='Dense4', downsample=False, keep_prob=FLAGS.dropout_factor)

    print('End Dims', conv)

    # Linear layers
    fc = sdn.fc7_layer('FC', conv, 16, True, phase_train, FLAGS.dropout_factor, BN=True)
    fc = sdn.linear_layer('Linear', fc, 8, False, phase_train, BN=True)
    Logits = sdn.linear_layer('Output', fc, FLAGS.num_classes, False, phase_train, BN=False, relu=False, add_bias=False)

    return Logits, sdn.calc_L2_Loss(FLAGS.l2_gamma)


def backward_pass(total_loss):

    """
    This function performs our backward pass and updates our gradients
    :param total_loss:
    :return:
    """

    # Get the tensor that keeps track of step in this graph or create one if not there
    global_step = tf.train.get_or_create_global_step()

    # Print summary of total loss
    tf.summary.scalar('Total_Loss', total_loss)

    # Define optimizer
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    # Compute the gradients
    gradients = opt.compute_gradients(total_loss)

    # Clilp the gradients
    gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]

    # Apply the gradients
    train_op = opt.apply_gradients(gradients, global_step, name='train')

    # Add histograms for the trainable variables. i.e. the collection of variables created with Trainable=True
    for var in tf.trainable_variables(): tf.summary.histogram(var.op.name, var)

    # Maintain average weights to smooth out training
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay, global_step)

    # Applies the average to the variables in the trainable ops collection
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # Does nothing. placeholder to control the execution of the graph
    with tf.control_dependencies([train_op, variable_averages_op]): dummy_op = tf.no_op(name='train')

    return dummy_op


def inputs(skip=False):

    """
    This function loads our raw inputs, processes them to a protobuffer that is then saved and
    loads the protobuffer into a batch of tensors
    """

    # To Do: Skip part 1 and 2 if the protobuff already exists
    if not skip: Input.pre_process(FLAGS.box_dims)

    print('----------------------------------------Loading Protobuff...')
    train = Input.load_protobuf()
    valid = Input.load_validation_set()


    return train, valid