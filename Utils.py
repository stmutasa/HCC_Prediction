
import os

import numpy as np
import tensorflow as tf
import SODLoader as SDL
import SOD_Display as Display
import pydicom as dicom
from medpy.io import save as mpsave
import matplotlib.pyplot as plt
import imageio
import cv2

from pathlib import Path
from random import shuffle
from Array2Gif import gif

# Define the flags class for variables
FLAGS = tf.app.flags.FLAGS

# Define the data directory to use
home_dir = str(Path.home()) + '/PycharmProjects/Datasets/HCC/'

# Utility classes
sdl = SDL.SODLoader(data_root=home_dir)
sdd = Display.SOD_Display()

def display_warps(box_dims=64, xforms=20):

    """
    We want to load all of the examples and perform affine warps, then display example images
    Dimensions to display
    number of warps to show
    """

    # Load the filenames and randomly shuffle them
    filenames = sdl.retreive_filelist('nii.gz', True, home_dir + 'HCC_registered/')
    shuffle(filenames)

    # Load labels
    labels = sdl.load_CSV_Dict('PT', home_dir+'labels.csv')

    # Global variables
    counter, data, counter2 = [0, 0], {}, [0, 0]
    index, pt, tracker = 0, 0, 0

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
        volume = np.concatenate([p1_vol, p2_vol, p3_vol], axis=-1).astype(np.float32)
        #volume = sdl.normalize_MRI_histogram(volumes)
        del p1_vol, p2_vol, p3_vol

        # Use the largest segments for the network
        segments, seg_size = [p1_seg, p2_seg, p3_seg], np.asarray([np.sum(p1_seg), np.sum(p2_seg), np.sum(p3_seg)])
        largest_seg = np.argmax(seg_size)
        segment = np.squeeze(segments[largest_seg])
        del p1_seg, p2_seg, p3_seg, segments, seg_size, largest_seg

        # Retreive the center of the segments and the "radius"
        blob, cn = sdl.largest_blob(segment)
        radius = min([int(np.sum(segment) ** (1/3) * 2.5), box_dims * 2])
        del blob

        # Save the images for this patient
        imgz = []
        first_box, _ = sdl.generate_box(volume[...,0], cn, radius)

        # Do the warped transforms
        for q in range(0, xforms):

            # The larger matrix we will be rotating to compensate for interpolations
            warp = np.copy(volume[...,0])
            warp_seg = np.copy(segment)

            # Now apply an affine transform to the volume
            warp_params = sdl.calc_fast_affine(cn, [10, 45,45])
            warp = sdl.perform_fast_affine(warp, warp_params)
            warp_seg = sdl.perform_fast_affine(warp_seg, warp_params)

            # Retreive the center
            _, ctr2 = sdl.largest_blob(warp_seg)

            # Box of the lesion
            box, new_center = sdl.generate_box(warp, ctr2, box_dims, z_overwrite=box_dims)
            norm_box, _ = sdl.generate_box(warp, ctr2, radius)

            # TODO: Testing: Displays
            # sdd.display_mosaic(warp)
            # sdd.display_single_image(warp[ctr2[0], :, :], False)
            # sdd.display_single_image(box[:, :, ctr2[2]])
            # sdd.display_single_image(box[8, :, :], False)
            # sdd.display_single_image(norm_box[radius//2, :, :])
            # sdd.display_single_image(box[:, ctr2[1], :])

            # Generate the images
            imgz.append(norm_box[radius//4, :, :])

            del warp, warp_seg, box, norm_box

        # Done with this patient
        pt +=1
        imgz.append(first_box[radius // 4, :, :])
        sdd.display_mosaic(imgz)
        if pt % 5 ==0:
            sdd.display_single_image(first_box[radius//4], True)
        del volume, first_box

    # Done with all files
    del data


def sort_files():

    """
    Sorts the DICOMs
    """

    # First define the filenames
    patient_list = sdl.retreive_filelist('*', path=home_dir+'Liver_Raw/', include_subfolders=True)
    shuffle(patient_list)
    ptc = 0

    # Loop through the patients
    for patient in patient_list:

        # Patients again
        patient_name = sdl.retreive_filelist('*', path=patient+'/', include_subfolders=True)

        for pt in patient_name:

            # Retreive the individual studies
            study_list = sdl.retreive_filelist('*', path=pt+'/', include_subfolders=True)
            shuffle(study_list)

            # Loop through the studies
            for study in study_list:

                # Retreive the series
                print ('\n\n')
                series_list = sdl.retreive_filelist('*', path=study+'/', include_subfolders=True)
                shuffle(series_list)
                ptc+=1

                # Loop through each series and load the DICOM
                for series in series_list:

                    # Load the DICOM
                    try: header, image = get_DICOM(series)
                    except:
                        print ('************* Unable to load header: ', series)
                        continue

                    Series = header['tags'].SeriesDescription
                    Accno = header['tags'].AccessionNumber
                    Study = header['tags'].StudyDescription
                    Manufacturer = header['tags'].Manufacturer
                    Time = header['tags'].AcquisitionTime

                    if 'SIEMENS' in Manufacturer: Maker = 'Siemens'
                    else: Maker = 'GE'

                    # Print. Sometimes the dynamics have the same name, load the time stamp too
                    print ('Patient: %s, Study: %s, Maker: %s, Series: %s - %s' %(Accno, Study, Maker, Series, Time))

                    # Filename
                    savedir = (home_dir + 'Processed2/' + Maker +'/')
                    savefile = savedir + (str(ptc) + '_' + Accno + '_' + Series +'_'+ Time)+'.nii.gz'

                    if not os.path.exists(savedir): os.makedirs(savedir)

                    mpsave(np.swapaxes(image, 0, 2), savefile)


def get_DICOM(path, sort=False, overwrite_dims=513, display=False):

    """
    This function loads a DICOM folder and stores it into a numpy array. From Kaggle
    :param: path: The path of the DICOM folder
    :param sort: Whether to sort through messy folders for the actual axial acquisition
    :param: overwrite_dims = In case slice dimensions can't be retreived, define overwrite dimensions here
    :param: dtype = what data type to save the image as
    :param: display = Whether to display debug text
    :param return_header = whether to return the header dictionary
    :return: image = A 3D numpy array of the image
    :return: numpyorigin, the real world coordinates of the origin
    :return: numpyspacing: An array of the spacing of the CT scanner used
    :return: spacing: The spacing of the pixels in millimeters
    :return header: a dictionary of the file's header information
    """

    # Some DICOMs end in .dcm, others do not
    if path[-3:] != 'dcm': fnames = [path + '/' + s for s in os.listdir(path) if s[-3:].lower() == 'dcm']
    else: fnames = [path]

    # Sort the slices
    ndimage = [dicom.read_file(path, force=True) for path in fnames]
    if sort: ndimage = sdl.sort_DICOMS(ndimage, display, path)
    ndimage, fnames, orientation, st, shape, four_d = sdl.sort_dcm(ndimage, fnames)
    ndimage.sort(key=lambda x: int(x.ImagePositionPatient[2]))

    # Retreive the dimensions of the scan
    try: dims = np.array([int(ndimage[0].Columns), int(ndimage[0].Rows)])
    except: dims = np.array([overwrite_dims, overwrite_dims])

    # Retreive the spacing of the pixels in the XY dimensions
    pixel_spacing = ndimage[0].PixelSpacing

    # Create spacing matrix
    numpySpacing = np.array([st, float(pixel_spacing[0]), float(pixel_spacing[1])])

    # Retreive the origin of the scan
    orig = ndimage[0].ImagePositionPatient

    # Make a numpy array of the origin
    numpyOrigin = np.array([float(orig[2]), float(orig[0]), float(orig[1])])

    # Finally, make the image actually equal to the pixel data and not the header
    try:
        image = np.stack([sdl.read_dcm_uncompressed(s) for s in ndimage])
    except:
        image = np.stack([sdl.read_dcm_compressed(f) for f in fnames])

    image = sdl.compress_bits(image)

    # Set image data type to the type specified
    image = image.astype(np.int16)

    # Convert to Houndsfield units
    if hasattr(ndimage[0], 'RescaleIntercept') and hasattr(ndimage[0], 'RescaleSlope'):
        for slice_number in range(len(ndimage)):
            intercept = ndimage[slice_number].RescaleIntercept
            slope = ndimage[slice_number].RescaleSlope

            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype('int16')
            image[slice_number] += np.int16(intercept)

    # --- Save first slice for header information
    header = {'orientation': orientation, 'slices': shape[1], 'channels': shape[0], 'fnames': fnames,
              'tags': ndimage[0], '4d': four_d, 'Origin': numpyOrigin, 'Spacing': numpySpacing, 'Dims': dims,}

    return header, image


def make_gifs():

    """
    Takes the input nifti files and saves .gif files of the volumes
    """

    # First define the filenames
    series_list = sdl.retreive_filelist('nii.gz', path=home_dir+'Processed2/', include_subfolders=True)
    shuffle(series_list)

    # Start from scratch
    if tf.io.gfile.exists('Gifs/'): tf.gfile.DeleteRecursively('Gifs')

    # Problem cases
    problems = [7, 20, 21, 26, 27, 31, 32, 44]

    # Vars
    index = 0

    # Loop through the patients
    for file in series_list:

        # Only do problem cases
        pt = int(os.path.basename(file).split('_')[0])
        if pt not in problems: continue

        # Retreive the filename
        save_file = 'Gifs/' + file.split('/')[-2] + '/' + file.split('/')[-1].split('.nii.gz')[-2] + '.gif'

        # Load the volume
        volume = np.squeeze(sdl.load_NIFTY(file))

        # Swapaxes for some reason
        volume = np.swapaxes(volume, 1, 2)

        # Normalize
        volume_norm = np.zeros_like(volume, dtype=np.uint8)
        for z in range (volume.shape[0]):
            volume_norm[z] = cv2.normalize(volume[z], dst=volume_norm[z], alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # File managment
        if not tf.io.gfile.exists(os.path.dirname(save_file)): tf.io.gfile.makedirs(os.path.dirname(save_file))

        # Save the .gif set FPS to volume depended
        fps = volume_norm.shape[0] //5
        gif(save_file, volume_norm, fps=fps, scale=0.8)

        index +=1
        del volume_norm, volume


def make_gifs_old_data():

    """
    Takes the input nifti files and saves .gif files of the volumes. This version is for the old, pre annotated cases
    We want to use the annotated names
    NOTE: reshuffle produced different patient IDs!!
    """

    # First define the filenames
    series_list = sdl.retreive_filelist('nii.gz', path=home_dir+'Processed2/', include_subfolders=True)
    annotations = sdl.retreive_filelist('jpg', path=home_dir+'Post-Annotated/', include_subfolders=True)
    seizures = sdl.retreive_filelist('gif', path=home_dir + 'Seizures/', include_subfolders=True)
    seizures = [x for x in seizures if 'INT_' in x]
    print (seizures)

    # Start from scratch
    if tf.io.gfile.exists('Gifs/'): tf.gfile.DeleteRecursively('Gifs')
    tf.io.gfile.mkdir('Gifs/')

    # Vars
    index = 0

    # Loop through the patients
    for file in series_list:

        # MRN and Time are unique to each patient:
        # index_MRN_Series_Time.nii.gz
        id = os.path.basename(file).split('_')[0] + '_'
        mrn = os.path.basename(file).split('_')[1]
        time = os.path.basename(file).split('_')[-1].split('.nii.gz')[0]
        prefix, series = 'RAW_', file.split('_')[-2]

        # Retreive this patients annotated name if available
        # Annotations are index_MRN_time_label
        for ann in annotations:
            if mrn in ann and time in ann:
                prefix, series = 'ANN_', ann.split('_')[-1].split('.jpg')[0]
                break

        # Retreive this patients INT status if done
        # Annotations are index_MRN_series_time
        for sez in seizures:
            if mrn in sez and time in sez:
                prefix = 'INT_'
                break

        # Save file as: Prefix_ID_MRN_Series_Time
        # Prefix = INT if interleaved, ANN if previously annotated, RAW if not previously annotated
        save_file = 'Gifs/' + prefix + id + mrn + '_' + series + '_' + time + '.gif'
        print (save_file, os.path.basename(file))

        # Load the volume
        volume = np.squeeze(sdl.load_NIFTY(file))

        # Swapaxes for some reason
        volume = np.swapaxes(volume, 1, 2)

        # Normalize
        volume_norm = np.zeros_like(volume, dtype=np.uint8)
        for z in range (volume.shape[0]):
            volume_norm[z] = cv2.normalize(volume[z], dst=volume_norm[z], alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # Save the .gif set FPS to volume depended
        fps = volume_norm.shape[0] //4
        gif(save_file, volume_norm, fps=fps, scale=1.0)

        index +=1
        del volume_norm, volume

#display_warps(56, 20)
#make_gifs()
make_gifs_old_data()