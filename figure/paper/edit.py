'''
edit nii file to create jpg files
jpg images will be used as figures in paper.

1. nrrd file read
2. nifti file read
3. edition process
4. jpg file save
'''

import os
import nrrd
import numpy as np
import SimpleITK as sitk
import matplotlib.image
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import label2rgb

def scale_intensity(array):
    """
    linear mapping the value into (0,1)
    """
    maxi = np.amax(array)
    mini = np.amin(array)
    norm_array = (array - mini) / (maxi - mini)
    # array = norm_array*255
    # return array.astype(dtype=np.int32)
    return norm_array

def box_count(array, bs):
    slide = bs
    shape = 560
    s1, s2 = shape//bs, shape//bs
    print(s1, s2, bs)
    box_count = 0
    grid_list = []
    for i in range(s1 + 1):
        for j in range(s2 + 1):
            box = array[i * slide:i * slide + bs, j * slide:j * slide + bs]
            pos = np.where(box)
            if len(pos[0]) or len(pos[1]):
                box_count += 1
                grid_list.append([i,j])
    return grid_list

def CV_count(array, bs):
    slide = bs
    shape = 560
    s1, s2 = shape//bs, shape//bs
    print(s1, s2, bs)
    box_count = 0
    grid_list = []
    cv_list = []
    for i in range(s1 + 1):
        for j in range(s2 + 1):
            box = array[i * slide:i * slide + bs, j * slide:j * slide + bs]
            pos = np.where(box)
            if len(pos[0]) or len(pos[1]):
                box_count += 1
                grid_list.append([i,j])
                avg = np.mean(box)
                std = np.std(box)
                cv = std / avg
                cv_list.append(cv)
    return grid_list, cv_list

def draw_box(grid_list, bs):
    color = 1
    slide = bs
    shape = 560
    grid = np.zeros((shape,shape))
    for coor in grid_list:
        x, y = coor
        # fill the pixels inside the box
        # grid[x*slide : x*slide+bs, y*slide : y*slide+bs] = color

        # draw only boundary pixels
        grid[x*slide : x*slide+bs, y*slide] = color
        grid[x*slide : x*slide+bs, y*slide+bs] = color
        grid[x*slide, y*slide : y*slide+bs + 1] = color
        grid[x*slide+bs, y*slide : y*slide+bs + 1] = color

        #thcik liner
        grid[x * slide: x * slide + bs + 1, y * slide + 1] = color
        grid[x * slide: x * slide + bs + 1, y * slide + bs + 1] = color
        grid[x * slide + 1, y * slide: y * slide + bs + 2] = color
        grid[x * slide + bs + 1, y * slide: y * slide + bs + 2] = color

        # grid[x * slide + 1, y * slide: y * slide + bs] = color
        # grid[x * slide + bs + 1, y * slide: y * slide + bs] = color
        # grid[x * slide - 1, y * slide: y * slide + bs] = color
        # grid[x * slide + bs - 1, y * slide: y * slide + bs] = color


    return grid

def draw_CV(grid_list, cv_list, bs):
    slide = bs
    shape = 560
    s1, s2 = shape//bs, shape//bs
    cv_grid = np.zeros((shape,shape))
    for i, coor in enumerate(grid_list):
        x, y = coor
        intensity = cv_list[i]
        # fill the pixels inside the box
        cv_grid[x*slide : x*slide+bs, y*slide : y*slide+bs] = intensity

        # draw only boundary pixels
        # cv_grid[x*slide : x*slide+bs, y*slide] = color
        # cv_grid[x*slide : x*slide+bs, y*slide+bs] = color
        # cv_grid[x*slide, y*slide : y*slide+bs] = color
        # cv_grid[x*slide+bs, y*slide : y*slide+bs] = color

    return cv_grid

def visual_boxcount(image, grid):
    image = np.expand_dims(image, axis=2)
    image = scale_intensity(image)
    image = np.concatenate([image, image, image], axis=2)

    print(np.shape(image), np.shape(grid))
    s1, s2 = np.shape(grid)
    for x in range(s1):
        for y in range(s2):
            if grid[x,y] != 0:
                image[x,y,0] = 1
    return image

def visual_lacunarity(image, cv_grid):
    image = np.expand_dims(image, axis=2)
    image = scale_intensity(image)
    image = np.concatenate([image, image, image], axis=2)

    print(np.shape(image), np.shape(cv_grid))
    s1, s2 = np.shape(cv_grid)
    for x in range(s1):
        for y in range(s2):
            if cv_grid[x,y] != 0:
                # image[x,y,:] = image[x,y,:]*cv_grid[x,y]

                margin = 1 - image[x,y,0]
                image[x,y,0] += margin*cv_grid[x,y]

                # if cv_grid[x,y] < image[x,y,1]:
                #     print(cv_grid[x,y] ,image[x,y,1])
                #     image[x, y, 0] = image[x,y,1]
                # else:
                #     image[x,y,0] = 1*cv_grid[x,y]

                # image[x,y,1] = 0
                # image[x,y,2] = 0
                # image[x,y,1] = 1*cv_grid[x,y]
                # image[x,y,2] = 1*cv_grid[x,y]
    return image

def main(subj):
    samples = {
        "7395931":["7395931_T1C_norm.nii.gz","7395931_T1C_norm-label.nrrd",43],
        "1791830":["1791830_T1C_norm.nii.gz","1791830_T1C_norm-label.nrrd",68],
        "5876859":["5876859_T1C_norm.nii.gz","5876859_T1C_norm-label.nrrd",74]
    }
    # ----------------------------- image read part -----------------------------  #
    dpath = "/home/soopil/Desktop/github/z_sampleData/ewha_brain_tumor/paper figure"
    flist = os.listdir(dpath)
    print(flist)

    sample = samples[subj]

    fname_nii = sample[0]
    fname_nrrd = sample[1]
    z_pos = sample[2]-1

    file_path = os.path.join(dpath, fname_nrrd)
    mask, header = nrrd.read(file_path)
    mask = np.array(mask)
    mask = np.swapaxes(mask, axis1=0, axis2=1)
    mask = scale_intensity(mask)
    mask = mask.astype(dtype = np.float32)
    # image shape : (560, 560, 100)
    print("label shape : ",mask.shape)

    file_path = os.path.join(dpath, fname_nii)
    itk_file = sitk.ReadImage(file_path)
    image = sitk.GetArrayFromImage(itk_file)
    image = np.swapaxes(image, axis1=0, axis2=2)
    image = np.swapaxes(image, axis1=0, axis2=1)
    image = np.flip(image, axis=0)
    mask = np.flip(mask, axis=0)
    image = scale_intensity(image)

    shape = mask.shape[0]
    slice_mask = mask[:, :, z_pos]
    slice_image = image[:, :,z_pos]
    # mask[40:shape - 40, 40:shape - 40, 36]
    slice = image[40:shape - 40, 40:shape - 40, z_pos]
    slice = scale_intensity(slice)
    slice = np.expand_dims(slice, axis=2)
    slice = np.concatenate([slice, slice, slice], axis=2)
    matplotlib.image.imsave('image.png', slice)
    # ----------------------------- edition process for lacunarity-----------------------------  #
    box_size = [2**i for i in range(1,10)]

    is_savegrid = False
    is_drawgrid = True

    for bs in box_size:
        grid_list, cv_list = CV_count(slice_mask, bs=bs)
        cv_list = scale_intensity(cv_list)
        # we need to increase the intensity for visualization
        cv_list = np.sqrt(cv_list)
        # cv_list = (cv_list+1)/2
        # for i, cv in enumerate(cv_list):
        #     if cv < 0.2:
        #         cv_list[i] = 0

        cv_grid = draw_CV(grid_list, cv_list, bs=bs)
        if is_savegrid:
            cv_grid = np.expand_dims(cv_grid, axis=2)
            cv_grid = np.concatenate([cv_grid, cv_grid, cv_grid], axis=2)
            matplotlib.image.imsave('cv_{}.png'.format(bs), cv_grid)

        elif is_drawgrid:
            # drawn_image = label2rgb(cv_grid, slice_image)
            drawn_image = visual_lacunarity(slice_image, cv_grid)
            drawn_image = drawn_image[40:shape - 40, 40:shape - 40, :]
            matplotlib.image.imsave('image_cv_{}.png'.format(bs), drawn_image)

        # if bs == 8:
        #     assert False
        # assert False
    # raise ()

    # ----------------------------- edition process for box count-----------------------------  #
    box_size = [2**i for i in range(0,10)]

    is_savegrid = False
    is_drawgrid = True

    for bs in box_size:
        print(bs)
        grid_list = box_count(slice_mask, bs=bs)
        slice_grid = draw_box(grid_list, bs=bs)
        if is_savegrid:
            slice_grid = np.expand_dims(slice_grid, axis=2)
            slice_grid = np.concatenate([slice_grid, slice_grid, slice_grid], axis=2)
            matplotlib.image.imsave('grid_{}.png'.format(bs), slice_grid)

        elif is_drawgrid:
            drawn_image = visual_boxcount(slice_image, slice_grid)
            drawn_image = drawn_image[40:shape - 40, 40:shape - 40, :]
            matplotlib.image.imsave('image_grid_{}.png'.format(bs), drawn_image)

        # assert False

    raise()
    # ----------------------------- change dimension for save-----------------------------  #
    print("image shape : ",image.shape)
    # img_size = 560
    # slice_image = image[:,:,36]
    print('expand dimension')
    slice_mask = np.expand_dims(slice_mask, axis=2)
    slice_image = np.expand_dims(slice_image, axis=2)
    slice_image = scale_intensity(slice_image)
    print(slice_mask.shape)
    print(slice_image.shape)

    print('change 1 to 3 channel')
    slice_mask = np.concatenate([slice_mask, slice_mask, slice_mask], axis=2)
    slice_image = np.concatenate([slice_image, slice_image, slice_image], axis=2)

    # ----------------------------- image save part -----------------------------  #
    print(slice_mask.shape)
    print(slice_image.shape)
    matplotlib.image.imsave('mask.png', slice_mask)
    matplotlib.image.imsave('image.png', slice_image)
    # raise()
    # ----------------------------- image read part -----------------------------  #
    # matplotlib.image.imsave('mask.png', slice_mask, cmap=plt.cm.gray)
    # matplotlib.image.imsave('image.png', slice_image, cmap=plt.cm.gray)



if __name__ == "__main__":
    # main("7395931")
    # main("1791830")
    main("5876859")