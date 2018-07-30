"""
For the columbia university hepatocellular cancer data
"""

import os

import numpy as np
import SODLoader as SDL
import SOD_Display as Display

from pathlib import Path
from random import shuffle

# Define the data directory to use
home_dir = str(Path.home()) + '/PycharmProjects/Datasets/HCC/'

# Utility classes
sdl = SDL.SODLoader(data_root=home_dir)
sdd = Display.SOD_Display()
disp = Display.SOD_Display()

def pre_process(chunks=2):

    """
    Loads the files to a pickle dictionary
    :param chunks: How many chunks of data to save
    :return:
    """

    # Load the filenames and randomly shuffle them
    filenames = sdl.retreive_filelist('nrrd', True, home_dir + 'segments/') + sdl.retreive_filelist('nrrd', True, home_dir + 'segments2/')
    shuffle(filenames)

    # Load labels
    labels = sdl.load_CSV_Dict('PT', home_dir+'labels.csv')

    # Global variables
    display, counter, data, index, pt, pts_loaded, images, tracker = [], [0, 0], {}, 0, 0, [], {}, 0

    # Special cases that failed segmentation
    error = ['27_DEL', '93_PV', '117_DEL', '97_PV', '71_DEL', '117_HPB', '97_HPB', '93_HPB',
             '117_PV', '100_PV', '112_DEL', '93_DEL', '27_DEL', '100_DEL', '97_DEL']

    total_volumes = len(filenames) // 2 - len(error)

    for file in filenames:

        # Skip label files. will load later
        if 'label' in file: continue
        if '.seg.' in file: continue

        # Retreive patient information
        basename = os.path.basename(file).upper()
        patient = basename.split(' ')[0]

        # Assign the correct phase (for loading the segments) and sequence (for saving)
        if 'ART' in basename or 'AP' in basename or 'PH1' in basename: phase, sequence = 'AP', 'd1'
        elif 'PORT' in basename or 'PV' in basename or 'PH2' in basename: phase, sequence = 'PV', 'd2'
        elif 'DEL' in basename or 'PH3' in basename: phase, sequence = 'DEL', 'd3'
        elif 'HPB' in basename or 'PH4' in basename: phase, sequence = 'HPB', 'd4'
        else:
            print ('Unable to retreive phase for: ', file, basename)
            continue

        try: label = int(labels[patient]['Label'])
        except:
            print ('Unable to load: ', file)
            continue

        # Skip the patients with corrupt segmentations
        test = ('%s_%s' % (patient, phase))
        if test in error: continue

        # Load/normalize image then find/load the segmentations. Remember we're loading a 4 part tuple with size, origin and spacing
        volume, io, isp, ish = sdl.load_nrrd_3D(file)
        for z in filenames:
            if 'label' not in z: continue
            if os.path.basename(z).upper().split(' ')[0] != patient: continue
            if phase in z: segments, sego, ssp, ssh = sdl.load_nrrd_3D(z)

        # If the segments are in sparse mode, detect this by scanning for small segment image size and fix them
        if segments.shape[-1] < 128:

            print ('Reshaping PT %s''s segments from %s to: %s. Result: ' %(patient, segments.shape, volume.shape), end='')

            # Retreive the |seg to img| shift and adjust for slice thickness (segments will always be inside image
            so = np.abs(io-sego)//ssp

            # Send a warning if for some reason the spacing of the segments and image don't match
            if ssp.all() != isp.all(): print ('Warning, spacing different for ', z, file)

            # Copy segments into a shifted array padded with zeros thats the same size as the source image
            segs = np.zeros(shape=ish, dtype=np.uint8)

            # Define the end boundaries of the new segments based on the shift. (accounting for the image index differences)
            endx = int(so[0] + ssh[2])
            endy = int(so[1] + ssh[1])
            endz = int(so[2] + ssh[0])

            # Copy the segments in to the zero array. Error handling code
            try: segs[int(so[2]):endz, int(so[1]):endy, int(so[0]):endx] = segments
            except:
                print ('Unable to broadcast... ', ssh, ssp, sego, io, so)

            # Second part of reshape statement
            print (segs.shape, 'starting at ', so)

            # Remake the segments
            del segments
            segments = np.copy(segs)
            del segs

        # Normalize the volume
        volume = sdl.normalize_MRI_histogram(volume, False, center_type='mean')

        """
        Prepare this patients data to save it to a custom pickle dictionary
        The volumes are numpy arrays. The rest of the indices are characters
        Accession numbers are more unique than MRNs. Save both. If both aren't available and every file is a unique patient then just duplicate.
        Sequence characters should be -'t1', 't2' etc with 'd1', 'd2', 'dn' etc for dynamic phases
        Please make sure your unique_pt_id is unique to all the datasets. Same for study ID. 
        MRN_Equivalent should be unique for each patient but some patients will have multiple studies.
        """

        # Create a unique patient ID (MRN). In this case it will be equal to the accession number as all HCCs ghere are unique
        pt_ID = 'HCC_' + str(patient)

        data[index] = {'volume': volume.astype(np.float16), 'segments': segments.astype(np.uint8),
                       'mrn': pt_ID, 'accession': pt_ID, 'sequence': sequence,'xy_spacing': str(isp[1]), 'z_spacing': str(isp[-1])}

        # Garbage
        del segments, volume
        index +=1
        if index % 20 == 0: print ('\n%s of %s volumes loaded... %s%% done\n' %(index, total_volumes, 100*index//total_volumes))

    # Save the dictionary in x chunks
    print ('Loaded %s volumes fully.' %len(data))
    split_dicts = sdl.split_dict_equally(data, chunks)
    for z in range (chunks): sdl.save_dict_pickle(split_dicts[z], ('data/intermediate%s' %(z+1)))

pre_process(2)