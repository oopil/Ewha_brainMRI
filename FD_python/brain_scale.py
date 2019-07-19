import os
import nrrd
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.filters import threshold_otsu, threshold_local

def read_MRI(img_path):
    print("file path : {}" .format(img_path))
    img_path_decoded = img_path #.decode() # what the fuck !!! careful about decoding
    itk_file = sitk.ReadImage(img_path_decoded)
    array = sitk.GetArrayFromImage(itk_file)
    print(array.shape, type(array))
    # array = np.expand_dims(array, 3)
    return array.astype(np.float32), itk_file

def save_nifti_file(array, itk_file, sample_dir_path, save_file_name):
    draw_file_path = os.path.join(sample_dir_path, save_file_name)
    draw_file = sitk.GetImageFromArray(array)
    # draw_file.CopyInformation(itk_file)
    sitk.WriteImage(draw_file, draw_file_path)
    print('saved the file : {}'.format(draw_file_path))

def check_mask_area():
    print('start MRI label check.')
    sample_dir_path = '/home/soopil/Desktop/github/brainMRI_classification/sample_image/ADDlabel'
    file_name_str = 'T1Label.nii.gz  aparc.DKTatlas+aseg.nii  aseg.auto.nii aparc+aseg.nii  aparc.a2009s+aseg.nii'
    file_name = [ e for e in file_name_str.split(' ') if e != '']
    print('label file : ',file_name)
    isp = True
    brain_file = 'brain.nii'
    label = 0 # ADD
    brain_file_path = os.path.join(sample_dir_path, brain_file)
    brain_array, itk_file = read_MRI(brain_file_path, label)

    new_file = file_name[1]
    new_file_path = os.path.join(sample_dir_path, new_file)
    # new_file_path = file
    label_array, itk_file = read_MRI(new_file_path, label)
    new_label_list = count_label_num(label_array)
    print(new_label_list)
    lh_cort, rh_cort = [],[]
    subcort = []
    for label in sorted(new_label_list):
        if label // 1000 == 1:
            lh_cort.append(label)
        elif label // 1000 == 2:
            rh_cort.append(label)
        else:
            subcort.append(label)
    empty_space_shape = [256 for i in range(3)]
    empty_space = np.zeros(empty_space_shape)
    draw_array = empty_space
    dilation_iter = 3
    for lh_label in subcort:
        if lh_label == 91 or (lh_label >= 10 and lh_label <= 30) and (lh_label not in (14, 15, 16,24)):
            label_mask = empty_space
            label_mask[np.where(label_array == lh_label)] = lh_label
            dilation_label_mask = ndimage.morphology.binary_dilation(label_mask, iterations=dilation_iter).astype(draw_array.dtype)
            draw_array = draw_array + dilation_label_mask
    for lh_label in lh_cort:
        label_mask = empty_space
        label_mask[np.where(label_array == lh_label)] = lh_label
        dilation_label_mask = ndimage.morphology.binary_dilation(label_mask, iterations=dilation_iter).astype(
            draw_array.dtype)
        draw_array = draw_array + dilation_label_mask

    # erase non - label pixel intensity
    brain_array[np.where(draw_array == 0)] = 0
    save_file_name = 'dilation_maksed_brain' + '.nii'
    save_nifti_file(brain_array, itk_file, sample_dir_path, save_file_name)

def main():
    sample_SINCHON = {
        'T1':'4045934_T1C.nii.gz',
        'T2':'4045934_T2.nii.gz',
        'mask':'4045934_T1C-label.nrrd'
    }
    sample_EWHA = {
        'T1':'10016344_T1C.nii.gz',
        'T2':'10016344_T2.nii.gz',
        'mask':'10016344_T1C-label.nrrd'
    }
    base_dir = '/home/soopil/Desktop/github/z_sampleData/ewha_brain_tumor/'
    sample_label = os.path.join(base_dir,sample_SINCHON['mask'])
    sample_T1 = os.path.join(base_dir,sample_SINCHON['T1'])
    sample_T2 = os.path.join(base_dir,sample_SINCHON['T2'])

    sample_label = os.path.join(base_dir,sample_EWHA['mask'])
    sample_T1 = os.path.join(base_dir,sample_EWHA['T1'])
    sample_T2 = os.path.join(base_dir,sample_EWHA['T2'])
    label, header = nrrd.read(sample_label)
    print(label.shape)
    print(header)
    shape = label.shape
    maxi = np.amax(shape)
    mini = np.amin(shape)
    scale = int(round(maxi/mini))
    print(maxi, mini, scale)

    T1, itk_T1 = read_MRI(sample_T1)
    # T2, itk_T2 = read_MRI(sample_T2)

    # ------------------------ label rescaling part ------------------------ #
    label = np.swapaxes(label, 0, 2)
    zoom_size = (256,256,256)
    # i need to check the label_zoom it has float value after resizing
    # T1_zoom = rescale(T1, 0.5)
    T1_zoom = resize(T1, zoom_size)
    # T1_zoom = resize(T1, (64,64,64))

    label_zoom = resize(label, zoom_size, anti_aliasing=False) * 1e5
    # print(label_zoom)
    # ------------------------ label thresholding part ------------------------ #
    th = 2
    label_zoom[np.where(label_zoom < th)] = 0
    label_zoom[np.where(label_zoom > th)] = 1

    # T1_zoom = np.kron(T1, np.ones(scale_array))
    # label_zoom = np.kron(label, np.ones(scale_array))
    # print(np.shape(label_zoom))
    # print(np.shape(T1_zoom))
    save_nifti_file(label_zoom, itk_T1, base_dir, '10016344_label_zoom.nii.gz')
    save_nifti_file(T1_zoom, itk_T1, base_dir, '10016344_T1_zoom.nii.gz')
    # save_nifti_file(T1_zoom, itk_T1, base_dir, 'T1_zoom_64.nii.gz')

    # np.swapaxes(x, 0, 1)
    # lac = lacunarity(data)
    # print(np.log(lac))


if __name__ == '__main__':
    main()