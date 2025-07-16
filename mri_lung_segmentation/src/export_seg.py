import argparse
import numpy as np
import os

from skimage.transform import rescale

import SimpleITK


def rescale_and_export(
    seg,
    nii_path,
    flip_flag
):
    """
    If images were scaled down for segmentation to work, rescale them and then export nii
    """
    print('Rescaling and exporting images...')

    im_sitk = SimpleITK.ReadImage(nii_path)
    im = SimpleITK.GetArrayFromImage(im_sitk)

    if flip_flag:
        seg = np.flip(seg,0)

    scale_factor = (im.shape[1]/seg.shape[1],im.shape[2]/seg.shape[2])

    # rescale here
    output_seg = np.zeros((seg.shape[0],int(seg.shape[1]*scale_factor[0]),int(seg.shape[2]*scale_factor[1])))
    for i_ind in range(seg.shape[0]):
        output_seg[i_ind, :, :] = rescale(seg[i_ind, :, :].astype(bool), scale_factor, order=0)

    output_seg[output_seg>0] = 1

    # Write out segmentation here
    output_seg_sitk = SimpleITK.GetImageFromArray(output_seg)
    output_seg_sitk.CopyInformation(im_sitk)
    SimpleITK.WriteImage(output_seg_sitk, os.path.join(os.path.dirname(nii_path),f'lungSeg_{os.path.basename(nii_path)}'))

    print('Done!')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Rescale and export segmentation",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-p', '--predictionPath',
        help='Path to the numpy file containing the prediction',
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        '-n', '--niftiPath',
        help='Path to a nii file containing the original images',
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        '-f', '--flip',
        help='Flag (boolean) to flip images in z',
        type=bool,
        default=True,
        required=False,
    )

    args = parser.parse_args()

    seg = np.load(args.predictionPath)

    rescale_and_export(
        seg=seg[...,0],
        nii_path=args.niftiPath,
        flip_flag=args.flip
    )
