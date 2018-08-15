"""
Does our loading and preprocessing of files to a protobuf
"""

import glob, os

import numpy as np
import tensorflow as tf
import SODLoader as SDL
import SOD_Display as Display
import matplotlib.pyplot as plt

from pathlib import Path
from random import shuffle

# Define the flags class for variables
FLAGS = tf.app.flags.FLAGS

# Define the data directory to use
home_dir = str(Path.home()) + '/PycharmProjects/Datasets/HCC/'

# Utility classes
sdl = SDL.SODLoader(data_root=home_dir)
sdd = Display.SOD_Display()


def save_segments(box_dims=256, slice_gap=1):

    """
    Loads the files to a protobuf with saved 3 channel co-registered volumes and segmentations
    :param box_dims: dimensions of the saved images
    :param slice_gap: The slice gap to use for 2.5 D
    :return:
    """

    # Load the filenames and randomly shuffle them
    filenames = sdl.retreive_filelist('nii.gz', True, home_dir + 'HCC_registered/')

    # Load labels
    labels = sdl.load_CSV_Dict('PT', home_dir+'labels.csv')

    # Global variables
    display, counter, data, pts_loaded, images = [], [0, 0], {}, [], {}
    size, index, pt, tracker = 0, 0, 0, 0

    for file in filenames:

        # Work only on the arterial phase
        if 'P1' not in file: continue

        # Retreive the labels
        basename = os.path.basename(file).split('.')[0]
        sa = basename.split('_')
        if len(sa) > 3: continue
        reg_name, id_name, phase = sa[0], sa[1], sa[2]
        hcc = int(labels[id_name]['Label'])

        # Get the filenames
        p2_file, p3_file = file[:-10] + '_P2Warped.nii.gz', file[:-10] + '_P3Warped.nii.gz'
        seg_base = os.path.dirname(file) + '/' + reg_name[1:] + '_' + id_name + '_'
        p1_lab = seg_base + 'P1_label.nii.gz'
        p2_lab, p3_lab = seg_base + 'P2_label.nii.gz', seg_base + 'P3_label.nii.gz'

        # Now load all the volumes and segments
        p1_vol, p1_seg = sdl.load_NIFTY(file), sdl.load_NIFTY(p1_lab)
        p2_vol, p3_vol = sdl.load_NIFTY(p2_file), sdl.load_NIFTY(p3_file)
        p2_seg, p3_seg = sdl.load_NIFTY(p2_lab), sdl.load_NIFTY(p3_lab)

        # Join the volumes along channel dimensions and normalize
        volumes = np.concatenate([p1_vol, p2_vol, p3_vol], axis=-1).astype(np.float32)
        volume = sdl.normalize_MRI_histogram(volumes)
        del p1_vol, p2_vol, p3_vol, volumes

        # Use the largest segments for the network
        segments, seg_size = [p1_seg, p2_seg, p3_seg], np.asarray([np.sum(p1_seg), np.sum(p2_seg), np.sum(p3_seg)])
        largest_seg = np.argmax(seg_size)
        segment = np.squeeze(segments[largest_seg])
        del p1_seg, p2_seg, p3_seg, segments

        # Now we have the volume and the segments. Make a dictionary and save the files
        for z in range(volume.shape[0]):

            # Calculate a scaled slice shift
            sz = slice_gap

            # Skip very bottom and very top of image
            if ((z - 3 * sz) < 0) or ((z + 3 * sz) > volume.shape[0]): continue

            # Label is easy, just save the slice
            data_label = sdl.zoom_2D(segment, (box_dims, box_dims))

            # Generate the empty data array
            data_image = np.zeros(shape=[5, box_dims, box_dims], dtype=np.float32)

            # Set starting point
            zs = z - (2 * sz)

            # Save 5 slices with shift Sz
            for s in range(5): data_image[s, :, :] = sdl.zoom_2D(volume[zs + (s * sz)].astype(np.float32), [box_dims, box_dims])

            # Save the dictionary:
            data[index] = {'image_data': data_image.astype(np.float16), 'label_data': data_label.astype(np.uint8), 'accno': basename,
                           'slice': z, 'mrn': id_name, 'shape_z': volume.shape[0], 'shape_xy': volume.shape[2], 'hcc': hcc}

            # Finished with this slice
            index += 1

            # Garbage collection
            del data_label, data_image

        # Finished with all of this patients slices
        pt += 1
        if pt % 10 == 0: print('%s Patients loaded (%s slices)' %(pt, index))

    # Finished with all patients
    print('%s Total Patients loaded, %s Total slices saved' % (pt, index))
    sdl.save_tfrecords(data, xvals=5, file_root='data/2.5D Segs')
    sdl.save_dict_filetypes(data[index - 1])

    del data


def save_examples(box_dims=256, warps=1):

    """
    Loads the files to a protobuf of warped image examples
    :param box_dims: dimensions of the saved images
    :param warps: The number of affine warps to apply
    :return:
    """

    # Load the filenames and randomly shuffle them
    filenames = sdl.retreive_filelist('nii.gz', True, home_dir + 'HCC_registered/')

    # Load labels
    labels = sdl.load_CSV_Dict('PT', home_dir+'labels.csv')

    # Global variables
    display, counter, data, pts_loaded, images = [], [0, 0], {}, [], {}
    size, index, pt, tracker = 0, 0, 0, 0

    for file in filenames:

        # Work only on the arterial phase
        if 'P1' not in file: continue

        # Retreive the labels
        basename = os.path.basename(file).split('.')[0]
        sa = basename.split('_')
        if len(sa) > 3: continue
        reg_name, id_name, phase = sa[0], sa[1], sa[2]
        hcc = int(labels[id_name]['Label'])

        # Get the filenames
        p2_file, p3_file = file[:-10] + '_P2Warped.nii.gz', file[:-10] + '_P3Warped.nii.gz'
        seg_base = os.path.dirname(file) + '/' + reg_name[1:] + '_' + id_name + '_'
        p1_lab = seg_base + 'P1_label.nii.gz'
        p2_lab, p3_lab = seg_base + 'P2_label.nii.gz', seg_base + 'P3_label.nii.gz'

        # Now load all the volumes and segments
        p1_vol, p1_seg = sdl.load_NIFTY(file), sdl.load_NIFTY(p1_lab)
        p2_vol, p3_vol = sdl.load_NIFTY(p2_file), sdl.load_NIFTY(p3_file)
        p2_seg, p3_seg = sdl.load_NIFTY(p2_lab), sdl.load_NIFTY(p3_lab)

        # Join the volumes along channel dimensions and normalize
        volumes = np.concatenate([p1_vol, p2_vol, p3_vol], axis=-1).astype(np.float32)
        volume = sdl.normalize_MRI_histogram(volumes)
        del p1_vol, p2_vol, p3_vol, volumes

        # Use the largest segments for the network
        segments, seg_size = [p1_seg, p2_seg, p3_seg], np.asarray([np.sum(p1_seg), np.sum(p2_seg), np.sum(p3_seg)])
        largest_seg = np.argmax(seg_size)
        segment = np.squeeze(segments[largest_seg])
        del p1_seg, p2_seg, p3_seg, segments

        # Find the center of all the blobs
        _, centers, _, blob_count  = sdl.all_blobs(segment)

        # Now we have the volume and the segments. Make a dictionary and save the files
        for z in range(warps):

            # Save the dictionary:
            data[index] = {'image_data': data_image.astype(np.float16), 'label_data': data_label.astype(np.uint8), 'accno': basename,
                           'slice': z, 'mrn': id_name, 'shape_z': volume.shape[0], 'shape_xy': volume.shape[2], 'hcc': hcc, 'warp':1}

            # Finished with this slice
            index += 1

            # Garbage collection
            del data_label, data_image

        # Finished with all of this patients slices
        pt += 1
        if pt % 10 == 0: print('%s Patients loaded (%s slices)' %(pt, index))

    # Finished with all patients
    print('%s Total Patients loaded, %s Total slices saved' % (pt, index))
    sdl.save_tfrecords(data, xvals=5, file_root='data/2.5D Segs')
    sdl.save_dict_filetypes(data[index - 1])

    del data


def load_protobuf():

    """
    Loads the protocol buffer into a form to send to shuffle
    :param 
    :return:
    """

    # Load all the filenames in glob
    filenames1 = glob.glob('data/*.tfrecords')
    filenames = []

    # Define the filenames to remove
    for i in range(0, len(filenames1)):
        if FLAGS.test_files not in filenames1[i]: filenames.append(filenames1[i])

    # Show the file names
    print('Training files: %s' % filenames)

    # Load the dictionary
    data = sdl.load_tfrecords(filenames, FLAGS.box_dims*2, tf.float32, channels=4)

    # Data Augmentation ------------------

    # Random contrast and brightness
    data['data'] = tf.image.random_brightness(data['data'], max_delta=2)
    data['data'] = tf.image.random_contrast(data['data'], lower=0.95, upper=1.05)

    # Random gaussian noise
    T_noise = tf.random_uniform([1], 0, 0.1)
    noise = tf.random_uniform(shape=[FLAGS.box_dims * 2, FLAGS.box_dims * 2, 4], minval=-T_noise, maxval=T_noise)
    data['data'] = tf.add(data['data'], tf.cast(noise, tf.float32))

    # Randomly rotate
    angle = tf.random_uniform([1], -0.45, 0.45)
    data['data'] = tf.contrib.image.rotate(data['data'], angle)

    # Random shear:
    rand = []
    for z in range(4): rand.append(tf.random_uniform([], minval=-0.15, maxval=0.15, dtype=tf.float32))
    data['data'] = tf.contrib.image.transform(data['data'], [1, rand[0], rand[1], rand[2], 1, rand[3], 0, 0])

    # Crop center
    data['data'] = tf.image.central_crop(data['data'], 0.55)

    # Then randomly flip
    data['data'] = tf.image.random_flip_left_right(tf.image.random_flip_up_down(data['data']))

    # Random crop using a random resize
    data['data'] = tf.random_crop(data['data'], [FLAGS.box_dims, FLAGS.box_dims, 4])

    # Display the images
    tf.summary.image('Train IMG', tf.reshape(data['data'][:, :, 0], shape=[1, FLAGS.box_dims, FLAGS.box_dims, 1]), 4)

    # Reshape image
    data['data'] = tf.image.resize_images(data['data'], [FLAGS.network_dims, FLAGS.network_dims])

    # # Randomly dropout the other channels
    # pvp = tf.cond(tf.squeeze(tf.random_uniform([1], 0, 1, dtype=tf.float32)) > 0.5,
    #                                 lambda: tf.multiply(data['data'][:,:,1], 0), lambda:  tf.multiply(data['data'][:,:,1], 1))
    # dlp = tf.cond(tf.squeeze(tf.random_uniform([1], 0, 1, dtype=tf.float32)) > 0.6,
    #                                 lambda: tf.multiply(data['data'][:, :, 2], 0), lambda:  tf.multiply(data['data'][:, :, 2], 1))
    #
    # # Concat the information
    # data['data'] = tf.concat([tf.expand_dims(data['data'][:, :, 0], -1), tf.expand_dims(pvp, -1),
    #                           tf.expand_dims(dlp, -1), tf.expand_dims(data['data'][:, :, 3], -1)], axis=-1)

    # Display the images
    tf.summary.image('Train IMG', tf.reshape(data['data'][:,:,0], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 4)

    # For now, only use the arterial or PVP
    data['data'] = tf.cond(tf.squeeze(tf.random_uniform([1], 0, 1, dtype=tf.float32)) > 0.5,
                                     lambda: tf.multiply(data['data'][:,:,1], 1), lambda:  tf.multiply(data['data'][:,:,0], 1))

    # Return data dictionary
    return sdl.randomize_batches(data, FLAGS.batch_size)


def load_validation_set():

    """
        Same as load protobuf() but loads the validation set
        :return:
    """

    # Use Glob here
    filenames1 = glob.glob('data/*.tfrecords')
    filenames = []

    # Retreive only the right filename
    for i in range(0, len(filenames1)):
        if FLAGS.test_files in filenames1[i]: filenames.append(filenames1[i])

    print('Testing files: %s' % filenames)

    # Load the dictionary
    data = sdl.load_tfrecords(filenames, FLAGS.box_dims*2, tf.float32, channels=4)

    # Crop center
    data['data'] = tf.image.central_crop(data['data'], 0.5)

    # Reshape image
    data['data'] = tf.image.resize_images(data['data'], [FLAGS.network_dims, FLAGS.network_dims])

    # Display the images
    tf.summary.image('Test IMG', tf.reshape(data['data'][:,:,0], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 4)

    # For now, only use the arterial or PVP
    data['data'] = data['data'][:, :, 0]

    return sdl.val_batches(data, FLAGS.batch_size)

save_segments()
# save_examples()