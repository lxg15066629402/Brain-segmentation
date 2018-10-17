# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os
import sys
import numpy as np
import nibabel as nib



def binary_dice3d(s,g):
    """
    dice score of 3d binary volumes
    inputs: 
        s: segmentation volume
        g: ground truth volume
    outputs:
        dice: the dice score
    """
    assert(len(s.shape)==3)
    [Ds, Hs, Ws] = s.shape
    [Dg, Hg, Wg] = g.shape
    assert(Ds==Dg and Hs==Hg and Ws==Wg)
    prod = np.multiply(s, g)
    s0 = prod.sum()
    s1 = s.sum()
    s2 = g.sum()
    dice = 2.0*s0/(s1 + s2 + 1e-10)
    return dice
    # dice = 2 TP /(和 + 1e-10 ---??) # 之间的差别

# 注释 
# def get_filename(set_name, case_idx, input_name, loc = ''):
#
#     pattern = '{0}/iSeg2017/iSeg-2017-{1}/subject-{2}-{3}.hdr'
#     return pattern.format(loc, set_name, case_idx, input_name)
#
#
# def get_set_name(case_idx):
# 	return 'Training' if case_idx < 11 else 'Testing'
#
# # Seg
# # GroudTruth
# def read_data(case_idx, input_name, loc = 'Seg'):
# 	set_name = get_set_name(case_idx)
#
# 	img_path = get_filename(set_name, case_idx, input_name, loc)
#
# 	return nib.load(img_path)
#
# def read_vol(case_idx, input_name, loc = '')
# 	image_data = read_data(case_idx, input_name, loc)
#
# 	return image_data.get_data()[:,:,:,0]



def dice_of_brats_data_set(s_folder, g_folder):
    dice_all_data = []
    # for i in range(1, 3):
    #     print(i)
        # s_name = os.path.join(s_folder, subject-{2}-{3}.hdr)
        # pattern = 's_folder/subject-{1}-label.hdr'
    s_name = 's_folder/subject-2-label.hdr'
    # s_name = pattern.format(i)

    # pattern = 'g_folder/subject-{1}-label.hdr'
    g_name = 'g_folder/subject-2-label.hdr'

    # g_name = pattern.format(i)


    seg = nib.load(s_name)
    gt = nib.load(g_name)

    s_volume = seg.get_data()[:,:,:,0]
    g_volume = gt.get_data()[:,:,:,0]
    dice_one_volume = []
    if(type_idx == 0):
        temp_dice = binary_dice3d(s_volume > 0, g_volume > 0)
        dice_one_volume = [temp_dice]
    elif(type_idx == 1): # tumor core
        s_volume[s_volume == 10] = 0
        g_volume[g_volume == 10] = 0
        temp_dice = binary_dice3d(s_volume > 0, g_volume > 0)
        dice_one_volume = [temp_dice]
        # ====  #
        # elif(type_idx == 2): # enhance core
            # s_volume[s_volume == 4] = 0
            # g_volume[g_volume == 4] = 0
            # temp_dice = binary_dice3d(s_volume > 0, g_volume > 0)
            # dice_one_volume = [temp_dice]
        # ====  #
    else:  # enhance tumor
        for label in [10, 150, 250]: # dice of each class
            temp_dice = binary_dice3d(s_volume == label, g_volume == label)
            dice_one_volume.append(temp_dice)
    dice_all_data.append(dice_one_volume)
    return dice_all_data
    
if __name__ == '__main__':
    s_folder = 'Seg'
    g_folder = '/media/cad/c5290228-7b88-43ba-809c-870ee6577d2c/data/lvxiaogang/iSeg2017-nic_vicorob-master/iSeg2017-nic_vicorob-master/Gt'
    test_types = ['CFS','GM', 'WM']
    for type_idx in range(3):  
        dice = dice_of_brats_data_set(s_folder, g_folder)
        dice = np.asarray(dice)
        dice_mean = dice.mean(axis = 0)
        dice_std  = dice.std(axis  = 0)
        test_type = test_types[type_idx]
        # np.savetxt(s_folder + '/dice_{0:}.txt'.format(test_type), dice)
        # np.savetxt(s_folder + '/dice_{0:}_mean.txt'.format(test_type), dice_mean)
        # np.savetxt(s_folder + '/dice_{0:}_std.txt'.format(test_type), dice_std)
        print('tissue type', test_type)
        if(test_type == 'all'):
            print('tissue label', [1, 2, 4])  # tissue
        print('dice mean  ', dice_mean)
        print('dice std   ', dice_std)
 
