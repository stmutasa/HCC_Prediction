"""
Does our loading and preprocessing of files to a protobuf
"""

import glob, time

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
disp = Display.SOD_Display()


def split_dict_equally(input_dict, chunks=2):

    """
    Function that splits a dictionary into equal chunks
    :param input_dict: input dictionary
    :param chunks: number of chunks
    :return: A list of dictionaries
    """

    # Create empty dictionaries list first
    return_list = [dict() for idx in range(chunks)]
    idx = 0

    # Save into each listed dictionary one at a time
    for k,v in input_dict.items():
        return_list[idx][k] = v

        # Loop back to beginning
        if idx < chunks-1: idx += 1
        else: idx = 0

    return return_list


def pre_process(box_dims=76, chunks=3):

    """
    Loads the files to a protobuf in two parts: 1. combine all phases for each pt, 2: save the boxes
    :param box_dims: dimensions of the saved images
    :param chunks: number of chunks to save/load
    :return:
    """

    time.sleep(0)

    # Load the filenames and randomly shuffle them
    filenames = sdl.retreive_filelist('nrrd', True, home_dir + 'segments/') + sdl.retreive_filelist('nrrd', True, home_dir + 'segments2/')
    shuffle(filenames)
    print (len(filenames), 'Base Files: ', filenames)

    # Load labels
    labels = sdl.load_CSV_Dict('PT', home_dir+'labels.csv')
    print(len(labels), 'Labels: ', labels)

    # Global variables
    display, counter, data, index, pt, pts_loaded, images, tracker = [], [0, 0], {}, 0, 0, [], {}, 0
    size = 0

    # for file in filenames:
    #
    #     # Skip label files. will load later
    #     if 'label' in file: continue
    #     if '.seg.' in file: continue
    #
    #     # Retreive patient information
    #     basename = os.path.basename(file).upper()
    #     patient = basename.split(' ')[0]
    #
    #     if 'ART' in basename or 'AP' in basename or 'PH1' in basename: phase = 'AP'
    #     elif 'PORT' in basename or 'PV' in basename or 'PH2' in basename: phase = 'PV'
    #     elif 'DEL' in basename or 'PH3' in basename: phase = 'DEL'
    #     elif 'HPB' in basename or 'PH4' in basename: phase = 'HPB'
    #     else:
    #         print ('Unable to retreive phase for: ', file, basename)
    #         continue
    #
    #     try: label = int(labels[patient]['Label'])
    #     except:
    #         print ('Unable to load: ', file)
    #         continue
    #
    #     # Load the image volumes and segmentations
    #     try: _ = images[patient]
    #     except: images[patient] = { 'AP': None, 'PV': None, 'DEL': None, 'HPB': None, 'Label': label, 'PT': patient,
    #                                 'AP_Seg': None, 'PV_Seg': None, 'DEL_Seg': None, 'HPB_Seg': None}
    #
    #     # Load/normalize image then find/load the segmentations. Remember we're loading a 4 part tuple with size, origin and spacing
    #     images[patient][phase], _, _, shape = sdl.load_nrrd_3D(file)
    #     # images[patient][phase] = sdl.normalize_MRI_histogram(images[patient][phase], False, center_type='mean')
    #     # print(images[patient][phase].dtype, images[patient][phase].nbytes)
    #     for z in filenames:
    #         if 'label' not in z: continue
    #         if os.path.basename(z).upper().split(' ')[0] != patient: continue
    #         if phase in z: images[patient][phase+'_Seg'] = sdl.load_nrrd_3D(z)[0]
    #
    #     tracker +=1
    #     try: size += images[patient][phase+'_Seg'].nbytes+images[patient][phase].nbytes
    #     except: print ('No segmentations! ', file)
    #     if tracker %10 ==0:
    #         print ('Processed %s volumes %s MB so far' %(tracker, size//1e6))


    # # Save the dictionary in 3 chunks
    # print ('Loaded %s patients fully.' %len(images))
    # split_dicts = split_dict_equally(images, chunks)
    # for z in range (chunks): sdl.save_dict_pickle(split_dicts[z], ('data/intermediate%s' %(z+1)))

    # Loop through all 3 files
    for batch in range(chunks):

        images = sdl.load_dict_pickle(('data/intermediate%s_pickle.p' %(batch+1)))
        print('Loaded %s patients' % len(images))

        for patient, dic in images.items():

            # Patient global variables
            pt_data = {'AP': None, 'PV': None, 'DEL': None, 'HPB': None}

            # Now process the sequences for this patient
            image_index = ['AP', 'PV', 'DEL', 'HPB']
            for phase, volume in dic.items():
                if phase not in image_index: continue

                # Convert to numpy array
                try: volume, segments = np.asarray(volume, np.int16), np.asarray(dic[phase+'_Seg'], np.int16)
                except: continue

                # Normalize volume and create a box
                segments, cn = sdl.largest_blob(segments)
                print (cn)
                disp.display_single_image(volume[cn[0]], False, patient)
                disp.display_single_image(segments[cn[0]], False, phase)
            #     volume = sdl.normalize_MRI_histogram(volume)
            #     blob, cn, sizes, num_blobs = sdl.all_blobs(dic[phase+'_Seg'])
            #     try:
            #         for z in range (1):
            #
            #             # Generate the box. Save a double box for rotations later
            #             box, _ = sdl.generate_box(volume[cn[z][0]], (cn[z][1], cn[z][2]), box_dims*2, dim3d=False)
            #
            #             # Save the data
            #             pt_data[phase] = box.astype(np.float32)
            #     except:
            #         print ('Box saving failed for: ', patient, phase)
            #
            # # Now work on saving the volume of the scan with channels
            # image_data = np.zeros(shape=(box_dims*2, box_dims*2, 4), dtype=np.float32)
            #
            # # Set the channels as follows. If the phase doesnt exist, keep the channel as zero
            # try: image_data[: ,: , 0] = pt_data['AP']
            # except: pass
            # try: image_data[ :, :, 1] = pt_data['PV']
            # except: pass
            # try: image_data[ :, :, 2] = pt_data['DEL']
            # except: pass
            # try: image_data[ :, :, 3] = pt_data['HPB']
            # except: pass
            #
            # # Save the data
            # data[index] = {'data': image_data, 'label': dic['Label'], 'pt': patient}
            # del volume, segments, image_data, box

            # Increment counter
            index += 1
            counter[dic['Label']] += 1

            # Done with this patient
            pt += 1
            if pt%5==0:
                print ('%s patients saved...' %pt)
                plt.show()

        # Done with all patients in this chunk
        print ('Made %s boxes from %s patients. Class counts: %s' %(index, pt, counter))
        plt.show()

        # # Save the data
        # sdl.save_tfrecords(data, 1, file_root=('data/HCC%s_4C_76' %(batch+1)))
        # if batch==0: sdl.save_dict_filetypes(data[0])
        # del data
        # data = {}


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

pre_process()