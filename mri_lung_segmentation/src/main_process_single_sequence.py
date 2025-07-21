""" usage: main_process_single_sequence.py [-h] [-i imagesDir] [-m modelDir] [-s size] [-c cutOff]

Perform Lung Segmentations and Calculate Lung Volume for One MRI Sequence

arguments:
  -h, --help            show this help message and exit.
  -i imagesDir, --imagesDir
                        Path to the folder with MRI sequence images. Required argument.
  -m modelDir, --modelDir
                        Path to the folder with the pre-trained model weights.
  -s size, --size
                        Size (value,value) of the image for predictions.
  -t cutOff, --cutOff
                        Threshold value for predictions.
"""

# NJS: use a different approach to sort filenames
import re
import glob

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from pydicom import dcmread
from matplotlib import pyplot as plt

from skimage.transform import rescale

import SimpleITK

import segmentation_utils
import lung_volume_utils
from DL_utils.model2D import unet2D

def load_dicoms(data_path):
    # Load DICOM MRI Sequence
    patient_image_array = []
    # NJS: Change the method used for sorting DICOMs
    sequence_names = glob.glob(os.path.join(data_path,'*.dcm'))
    sequence_names_sorted = sorted(sequence_names,key=lambda s: int(re.findall(r'\d+',s)[-1]))

    for i in sequence_names_sorted:
        dicom_sequence = dcmread(i)
        image_array = dicom_sequence.pixel_array
        # add channels dim
        image_array_tf = tf.expand_dims(image_array, axis=-1)
        patient_image_array.append(image_array_tf)

    patient_image_array = np.array(patient_image_array)

    # Load DICOM Metadata
    meta_dict = {
            "x_spacing" : dicom_sequence.PixelSpacing[0],
            "y_spacing" : dicom_sequence.PixelSpacing[1],
            "slice_thickness" : dicom_sequence.SliceThickness,
            "spacing_between_slices" : dicom_sequence.SpacingBetweenSlices
            }

    return patient_image_array, meta_dict

def load_image(data_path):
    im_sitk = SimpleITK.ReadImage(data_path)
    image_array = SimpleITK.GetArrayFromImage(im_sitk)

    patient_image_array = []
    for i_ind in range(image_array.shape[0]):
        # add channels dim
        image_array_tf = tf.expand_dims(image_array[i_ind,:,:], axis=-1)
        patient_image_array.append(image_array_tf)

    patient_image_array = np.array(patient_image_array)

    meta_dict = {
            "x_spacing" : im_sitk.GetSpacing()[0],
            "y_spacing" : im_sitk.GetSpacing()[1],
            "slice_thickness" : im_sitk.GetSpacing()[2],
            "spacing_between_slices" : im_sitk.GetSpacing()[2]
            }
    return patient_image_array, meta_dict


def segmentation_volume_pipeline(
    data_path,
    patient_id,
    output_dir,
    models_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'all_pat'),
    size=128,
    cut_off=0.5,
    output_measures=False
):
    # NJS: check if path to image or directory of DICOMs
    if os.path.isdir(data_path):
        # Assume this is the path to DICOMs
        patient_image_array, meta_dict = load_dicoms(data_path)
    elif os.path.isfile(data_path):
        # Assume mha or nii
        patient_image_array, meta_dict = load_image(data_path)

    # Create Empty model
    segmentation_models = [unet2D(None, (size, size, 1), 1, "binary_crossentropy")]*3

    # Load Model Weights
    all_model_paths = []
    for file in os.listdir(models_path):
        if file.endswith(".hdf5"):
            all_model_paths.append(os.path.join(models_path, file))

    for idx, model in enumerate(segmentation_models):
        model.load_weights(all_model_paths[idx])

    # Standardize Sequence
    standardized_images = segmentation_utils.standardize_array_per_img(patient_image_array)
    assert np.max(standardized_images) == 1
    assert np.min(standardized_images) == 0

    #cropped_images = segmentation_utils.crop_patient_images(
    #    standardized_images[:, :, :, 0],
    #    target_shape=size,
    #)
    # NJS: Try resampling instead of cropping
    scale_factor = standardized_images.shape[1]/size
    cropped_images = np.zeros((patient_image_array.shape[0],size,size))
    for im in range(patient_image_array.shape[0]):
        cropped_images[im, :, :] = rescale(standardized_images[im, :, :, 0], (1/scale_factor, 1/scale_factor))

    cropped_images_tf = tf.expand_dims(cropped_images, axis=-1)

    prediction = segmentation_utils.gen_maj_pred_of_images(segmentation_models, cropped_images_tf, cut_off)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Save predictions as npy array
    with open(os.path.join(output_dir, f'{patient_id}.npy'), 'wb') as f:
        np.save(f, prediction[...,0])

    # If mha or nii, save prediction in the same format
    if os.path.isfile(data_path):
        # rescale here
        output_seg = np.zeros(standardized_images[...,0].shape)
        for i_ind in range(output_seg.shape[0]):
            output_seg[i_ind, :, :] = rescale(prediction[i_ind, :, :, 0].astype(bool), scale_factor, order=0)

        output_seg[output_seg>0] = 1

        # Write out segmentation here
        im_sitk = SimpleITK.ReadImage(data_path)
        output_seg_sitk = SimpleITK.GetImageFromArray(output_seg)
        output_seg_sitk.CopyInformation(im_sitk)
        SimpleITK.WriteImage(output_seg_sitk, os.path.join(output_dir,f'lungSeg_{os.path.basename(data_path)}'))

    # Optional: Plot predicted masks, calculate statistics and save pngs
    if output_measures:
        assert len(cropped_images)==len(prediction)
        for i in range(len(cropped_images)):
            fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
            axs[0].imshow(cropped_images[i, :, :], cmap='bone', aspect='auto')
            axs[1].imshow(prediction[i, :, :, 0], cmap='bone', aspect='auto')
            plt.savefig(os.path.join(output_dir, f'{patient_id}_{i:02d}.png'), dpi=100)
            plt.close()

        # Calculate volume
        object_3d = lung_volume_utils.gen_3d_object_from_numpy(prediction, meta_dict["slice_thickness"], meta_dict["spacing_between_slices"])
        left_volume, right_volume, silhouette = lung_volume_utils.calculate_left_and_right_volume(object_3d, meta_dict["x_spacing"], meta_dict["y_spacing"])
        overall_volume = lung_volume_utils.calculate_volume(object_3d, meta_dict["x_spacing"], meta_dict["y_spacing"])

        # Save extracted features:
        feature_df = pd.DataFrame({
            'patient_id': [patient_id],
            'left_vol': [left_volume],
            'right_vol': [right_volume],
            'overall_vol': [overall_volume]
        })
        feature_df.to_csv(os.path.join(output_dir, f'{patient_id}.csv'), index=False)

        print('left_volume', left_volume)
        print('right_volume', right_volume)
        print('overall_volume', overall_volume)


def main():
    # Initiate argument parser
    parser = argparse.ArgumentParser(description="Perform Segmentations and Calculate Volume for One MRI Sequence",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--imageSrc',
        help='Path to mha / nii file, or folder with DICOM data',
        type=str,
        default=None,
        required=True,
    )
    # NJS: add patientID
    parser.add_argument(
        '-p', '--patientID',
        help='ID of the patient',
        type=str,
        default=None,
        required=True,
    )
    # NJS: add outputDir optional argument
    parser.add_argument(
        '-o', '--outputDir',
        help='Output directory - will be the same as input directory by default',
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        '-m', '--modelDir',
        help='Path to the folder with the pre-trained model weights.',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..', 'models', 'all_pat'),
        required=False,
    )
    parser.add_argument(
        '-s', '--size',
        help='Image size, taken as (value,value) pixels for predictions',
        default=128,
        type=int)
    parser.add_argument(
        '-c', '--cutOff',
        help='Threshold value for predictions',
        default=0.5,
        type=int)
    # NJS: add optional argument to disable production of PNGs and output metrics
    parser.add_argument(
        '-t', '--outputMeasures',
        help='Flag to output PNGs of the prediction for each slice',
        default=False,
        required=False,
        type=bool)
    args = parser.parse_args()

    # Segment the lung and calculate the volumes
    segmentation_volume_pipeline(
        data_path=args.imageSrc,
        patient_id=args.patientID,
        output_dir=args.outputDir,
        models_path=args.modelDir,
        size=args.size,
        cut_off=args.cutOff,
        output_measures=args.outputMeasures
    )


if __name__ == '__main__':
    main()


# Usage example with args:
# python feature_extraction_and_regression/main_process_single_sequence.py -i 'data/BPD/sample/' -m 'models/all_pat' -sp 'results/BPD/sample/'
