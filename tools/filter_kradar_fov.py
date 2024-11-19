import os, sys
import cv2, imageio
import numpy as np
import torch
import time
from argparse import ArgumentParser
import yaml
from tqdm import tqdm


def main():

    root_path = "/mnt/data/DataSet/K-RadarOOC/"
    config_path = './config.yaml'

    parse = ArgumentParser()
    args=parse.parse_args()
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    voxel_size = config['voxel_size']
    pc_range = config['pc_range']
    occ_size = config['occ_size']

    # process one clip once
    splits = ['train']
    for split in splits:

        clips = sorted(os.listdir(os.path.join(root_path, split)))
        for clip in clips:
          if clip.isnumeric():
            print('***Start to process {}***'.format(clip))
            path = os.path.join(root_path, split, clip)
            occ_path = os.path.join(path,'semantic_occupancy_gt/')
            if not os.path.exists(occ_path):
                print(occ_path)
                continue
            save_path = os.path.join(path, 'semantic_occupancy_gt_fov/')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            len_sequence = len(os.listdir(occ_path))
            for i in tqdm(range(len_sequence)):
                voxels = np.load(os.path.join(occ_path, 'occupancy_gt_with_semantic{}.npy'.format(i)))
                # transfer from voxel coordinates to Cartersian coordinates
                cart_voxels = np.zeros((voxels.shape))
                cart_voxels[:,0] = pc_range[0] + voxels[:,0] * voxel_size
                cart_voxels[:,1] = pc_range[1] + voxels[:,1] * voxel_size
                cart_voxels[:,2] = pc_range[2] + voxels[:,2] * voxel_size
                cart_voxels[:,3] = voxels[:,3]
                # transfter to radar coordinates and compute the polar 
                cart_voxels_radar = cart_voxels + np.array([[-2.54, 0.3, 0.7, 0]])
                # radar_range_mask = (np.linalg.norm(cart_voxels_radar[:,:3], axis=1) < 118)
                radar_azimuth_mask =  (np.abs(np.degrees(np.arctan2(cart_voxels_radar[:,1], cart_voxels_radar[:,0])))<53.5)
                radar_mask = radar_azimuth_mask
                # filter voxels based on radar fov
                cart_voxels = voxels[radar_mask,:]
                filtered_voxels = cart_voxels[cart_voxels[:, 2] >= 6]
                filtered_voxels[:, 2] -= 6
                
                


                np.save(save_path + 'occupancy_gt_with_semantic_fov{}.npy'.format(i), filtered_voxels)
        




if __name__ == '__main__':
    main()
