
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


def save_accessions():

    """
    Save a list of accession numbers
    :return:
    """

    # First define the filenames
    series_list = sdl.retreive_filelist('nii.gz', path=home_dir + 'Processed2/', include_subfolders=True)

    # Vars
    index, list, data = 0, [], {}

    # Retreive the filename
    save_file = 'data/patient_list.csv'

    # Loop through the patients
    for file in series_list:

        # Only do problem cases
        pt = int(os.path.basename(file).split('_')[0])
        acc = int(os.path.basename(file).split('_')[1])

        if acc in list: continue

        list.append(acc)
        data[pt] = acc

    sdl.save_Dict_CSV(data, save_file)


def make_gifs_again():

    """
    Takes the input nifti files and saves .gif files of the volumes. This version uses the ones annotated before
    discovery of the SAM glitch
    """

    """
    Series list contains the original preprocessed files
    rechecks contains the originally annotated GIFS
    redownloads list contains the files lina retreived

    Load redownloads and save all those, if annotated use annotation
    Load series_list and get timestamp and accession
    if the file accession is in the dicom folder from lina, (generate list of these beforehand) use that instead
    if set filename equivalent to what was done before in the rechecks
    order filename as ID_ANN_ACC_SERIES DESC_TIME
    """

    # First retreive lists of the the filenames
    series_list = sdl.retreive_filelist('nii.gz', path=home_dir+'Processed2/', include_subfolders=True)
    rechecks = sdl.retreive_filelist('gif', True, path=home_dir+'Rechecks')
    redownloads = list()
    for (dirpath, dirnames, filenames) in os.walk(home_dir+'DICOM/'):
        redownloads += [os.path.join(dirpath, dir) for dir in dirnames]
    redownloads = [x for x in redownloads if 'OBJ_0' in x]

    # Start from scratch, delete the folder
    if tf.io.gfile.exists('Gifs/'): tf.gfile.DeleteRecursively('Gifs')
    tf.io.gfile.mkdir('Gifs/')

    # Vars
    index = 0

    # Loop through and generate accession list of redownloads
    re_accnos = ['4315499', '3384278', '3408296', '3604840', '3928391', '4934558', '3920797', '3635146', '4253405',
                 '4265735', '4390034', '3440998', '3514539', '4016397', '3439696', '3961248', '4250322', '4221344',
                 '6044067', '3820833', '3452961', '3401206', '4029113', '3838525', '3560271', '4264491', '3734178', '3329640']
    # for file in redownloads:
    #
    #     # Try and load the volume, don't load non dicoms
    #     try: volume, header = sdl.load_DICOM_3D(file, return_header=True)
    #     except: continue
    #
    #     # Retreive the tags we want
    #     Series = header['tags'].SeriesDescription
    #     Accno = header['tags'].AccessionNumber
    #     Time = header['tags'].AcquisitionTime
    #     Prefix = 'RAW'
    #
    #     # Get the index of the file, 68 + patient number
    #     ID = int(file.split('/')[-4].split('_')[1]) + 68
    #
    #     # If used in annotation, use that annotation
    #     for check in rechecks:
    #         if Accno in check and Time in check:
    #             if 'INT' in check:
    #                 Prefix = 'INT_' + check.split('/')[-1].split('_')[1]
    #                 Series = check.split('/')[-1].split('_')[4]
    #             else:
    #                 Prefix = check.split('/')[-1].split('_')[0]
    #                 Series = check.split('/')[-1].split('_')[3]
    #             break
    #
    #     # Remove '/' from series name to prevent saving errors
    #     new_Series = Series.replace('/', ' ')
    #     new_Series, Series = Series, new_Series
    #
    #     # Save the gif, first define filename: ID_ANN_ACC_SERIES DESC_TIME
    #     savefile = 'Gifs/' + str(ID) + '_' + Prefix + '_' + Accno + '_' + Series + '_' + Time
    #     print('Patient: %s, Series: %s - %s, File: %s' % (Accno, Series, Time, savefile))
    #     sdl.save_gif_volume(volume, savefile)
    #
    #     index += 1
    #     del volume


    # Loop through the patients
    for file in series_list:

        # MRN and Time are unique to each patient
        # 3_4979417_AX VIBE PRE_111725.605000
        base = os.path.basename(file)
        ID = base.split('_')[0] + '_'
        Accno = base.split('_')[1]
        Time = base.split('_')[-1].split('.')[0]
        Prefix, Series = 'RAW' , file.split('_')[-2]
        should_pass = None

        # If this is one of the redownloaded files by accession, skip everything past here
        for check in re_accnos:
            if Accno in check:
                should_pass = 1

        if should_pass: continue

        # if set filename equivalent to what was done before in the rechecks, use that
        for check in rechecks:
            if Accno in check and Time in check:
                if 'INT' in check:
                    Prefix = 'INT_' + check.split('/')[-1].split('_')[1]
                    Series = check.split('/')[-1].split('_')[4]
                else:
                    Prefix = check.split('/')[-1].split('_')[0]
                    Series = check.split('/')[-1].split('_')[3]
                break

        # Load the volume
        volume = np.squeeze(sdl.load_NIFTY(file))
        # Swapaxes for nifti files
        volume = np.swapaxes(volume, 1, 2)

        # Save the gif, first define filename: ID_ANN_ACC_SERIES DESC_TIME
        savefile = 'Gifs/' + str(ID) + '_' + Prefix + '_' + Accno + '_' + Series + '_' + Time
        print('Patient: %s, Series: %s - %s, File: %s' % (Accno, Series, Time, savefile))
        sdl.save_gif_volume(volume, savefile)

        index += 1
        del volume


def separate_DICOMs():

    """
    Helper function to separate interleaved DICOMs
    :return:
    """

    # First retreive lists of the the filenames
    interleaved = sdl.retreive_filelist('gif', path=home_dir+'Recheck 2/', include_subfolders=True)
    interleaved = [x for x in interleaved if '_INT_' in x]
    shuffle(interleaved)
    redownloads = list()
    for (dirpath, dirnames, filenames) in os.walk(home_dir + 'DICOM/'):
        redownloads += [os.path.join(dirpath, dir) for dir in dirnames]
    redownloads = [x for x in redownloads if 'OBJ_0' in x]
    shuffle (redownloads)

    # Load the redownloads and filter them
    for folder in redownloads:

        # First just load the headers to save time
        header = sdl.load_DICOM_Header(folder, multiple=True)
        try:
            Series = header['tags'].SeriesDescription
            Accno = header['tags'].AccessionNumber
            Study = header['tags'].StudyDescription
            Time = header['tags'].AcquisitionTime
        except: continue

        # Check header info with the labeled gifs
        study = [x for x in interleaved if Accno in x.split('_')[-3] and Time in x.split('_')[-1].replace('.gif', '')]
        if not study: continue

        # Now load the full study
        fnames = list()
        for (dirpath, dirnames, filenames) in os.walk(folder):
            fnames += [os.path.join(dirpath, file) for file in filenames]
        ndimage = [dicom.read_file(path, force=True) for path in fnames]

        """
         TODO: Sort the slices
         In and out of phase can be sorted by EchoTime
         They can be sorted by SliceLocation (check for duplicates)
         InstanceNumber doesn't work for in and out of phase (TE does)
        """

        # Make list with slice location and check how many repeated positions there are
        sort_list = np.asarray([x.SliceLocation for x in ndimage], np.int16)
        unique, counts = np.unique(sort_list, return_counts=True)
        repeats = np.max(counts)

        # Sort the images by ImagePositionPatient
        ndimage, _, _, _, _, _ = sdl.sort_dcm(ndimage, fnames)
        ndimage.sort(key=lambda x: int(x.ImagePositionPatient[2]))

        # Loop through the number of repeats and create that many volumes
        for r in range(repeats):

            # Create array with every xx slice
            slice_subset = []
            for i in range(r, len(ndimage), repeats): slice_subset.append(ndimage[i])

            # Make the image actually equal to the pixel data and not the header
            try: image = np.stack([sdl.read_dcm_uncompressed(s) for s in slice_subset])
            except:
                print ('Cant load: ', study[0])
                continue
            image = sdl.compress_bits(image)
            image = image.astype(np.int16)

            # Convert to Houndsfield units
            if hasattr(slice_subset[0], 'RescaleIntercept') and hasattr(slice_subset[0], 'RescaleSlope'):
                for slice_number in range(len(slice_subset)):
                    intercept = slice_subset[slice_number].RescaleIntercept
                    slope = slice_subset[slice_number].RescaleSlope

                    image[slice_number] = slope * image[slice_number].astype(np.float64)
                    image[slice_number] = image[slice_number].astype('int16')
                    image[slice_number] += np.int16(intercept)

            # Now image has our volume
            volume = np.asarray(image)

            # Get savefile name
            save_gif = study[0].replace('_INT', ('-%s' %r))
            save_gif = save_gif.replace('Recheck 2', 'INT_Fixed')
            save_vol = save_gif.replace('.gif', '.nii.gz')

            # Save the gif and volume
            print ('Saving: ', os.path.basename(save_vol))
            sdl.save_volume(volume, save_vol, compress=True)
            sdl.save_gif_volume(volume, save_gif)

        #     # TODO:
        #     sdd.display_volume(volume)
        # plt.show()


def separate_DICOMs2():

    """
    Helper function to separate interleaved DICOMs
    For the cornell DICOMS
    :return:
    """

    # First retreive lists of the the filenames
    interleaved = sdl.retreive_filelist('gif', path=home_dir+'Recheck 2/', include_subfolders=True)
    interleaved = [x for x in interleaved if '_INT_' in x]
    shuffle(interleaved)
    cornell2 = sdl.retreive_filelist('nii.gz', True, home_dir+'Raw2_Processed/Siemens/')
    shuffle (cornell2)
    index = 1
    accnos3 = []

    # Load the redownloads and filter them
    for file in cornell2:

        # Get accno and time
        base = os.path.basename(file)
        Accno = base.split('_')[1]
        Time = base.split('_')[-1].split('.')[0]

        # Check header info with the labeled gifs
        study = [x for x in interleaved if Accno in x.split('_')[-3] and Time in x.split('_')[-1].replace('.gif', '')]
        if not study: continue

        # Now load the full study
        volume_int = sdl.load_NIFTY(file)

        # Calculate the number of repeats for Siemens (Cornell) studies
        # DWI's repeat 3 times or once, Subs are 2 (obv), dynamics are 3, in/out are 2
        repeats = 2
        if 'DYN X3' in base.upper() and 'SUB' not in base.upper(): repeats = 3
        if 'DWI' in base.upper(): repeats = 3
        if 'DWI' in base.upper() and volume_int.shape[0] <= 51 : repeats = 1

        """
         TODO: Sort the slices
         In and out of phase can be sorted by EchoTime
         They can be sorted by SliceLocation (check for duplicates)
         InstanceNumber doesn't work for in and out of phase (TE does)
        """

        # Loop through the number of repeats and create that many volumes
        for r in range(repeats):

            # Create array with every xx slice
            slice_subset = []
            for i in range(r, volume_int.shape[0], repeats):
                slice_subset.append(volume_int[i])

            volume = np.asarray(slice_subset)

            # Get savefile name
            save_gif = study[0].replace('_INT', ('-%s' %r))
            save_gif = save_gif.replace('Recheck 2', 'INT_Fixed/gif')
            save_vol = save_gif.replace('.gif', '.nii.gz')
            save_gif = save_gif.replace('gif/', 'vol/')

            # Save the gif and volume
            print ('Saving: ', os.path.basename(save_vol))
            sdl.save_volume(volume, save_vol, compress=True)
            sdl.save_gif_volume(volume, save_gif)

            # TODO: Testing
            # print(index, '-----', volume.shape, '->', volume_int.shape, '----', base)
            sdd.display_volume(volume)
        index +=1
        #plt.show()
    plt.show()


#display_warps(56, 20)
#make_gifs()
#make_gifs_old_data()
#make_gifs_again()
separate_DICOMs2()