import os
from multiprocessing import Pool

import mmcv
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
import copy

# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py#L834
def map_pointcloud_to_image(
    pc,
    im,
    cam_intrinsic,
    min_dist: float = 0.0,
):
    points = np.asarray(pc[:,:3])
    points = points[points[:,0]>min_dist]
    ones = np.ones((points.shape[0], 1))
    homo_points = np.hstack((points, ones)).T
    # Append ones to convert to homogeneous coordinates for transformation
    # Transformation matrix (assuming it's fixed as in your initial code)
    transform_matrix = np.array([
        [6.29238090e+02, -5.27036820e+02, -4.59938064e+00, 1.41321630e+03],
        [3.72013478e+02,  9.42970300e+00, -5.59685543e+02, 1.07637845e+03],
        [9.99925370e-01,  1.22165356e-02,  1.06612091e-04, 1.84000000e+00]
    ])

    # Apply transformation

    uvs = (transform_matrix @ homo_points).T

    # Convert to non-homogeneous coordinates
    uv = uvs[:, :2] / uvs[:, 2, np.newaxis]
    depths = uvs[:, 2]

    # Initialize mask based on depth values being greater than min_dist
    mask = depths > min_dist

    # Update mask based on uv coordinates being within the image boundaries
    mask = np.logical_and(mask, uv[:, 0] > 1)
    mask = np.logical_and(mask, uv[:, 0] < im.shape[1] - 1)  # Image width boundary
    mask = np.logical_and(mask, uv[:, 1] > 1)
    mask = np.logical_and(mask, uv[:, 1] < im.shape[0] - 1)  # Image height boundary

    # Use the mask to filter points and depths
    points_valid = uv[mask]
    depths_valid = depths[mask]

    return points_valid, depths_valid


info_path_train = '/home/xiangyu/SurroundOcc/kradar_dict_train.pkl'
info_path_val = '/home/xiangyu/SurroundOcc/kradar_dict_val.pkl'
lidar_key = 'LIDAR_TOP'
cam_keys = [
'CAM_FRONT'
]
def worker(info):
    lidar_path = info['lidar_path']
    points = np.fromfile(lidar_path,
                         dtype=np.float32,
                         count=-1).reshape(-1, 5)[..., :4]
    
    for i, cam_key in enumerate(cam_keys):
        cam_intrinsic = info['cams'][cam_key]['cam_intrinsic']
        img = mmcv.imread(
            os.path.join(info['cams'][cam_key]['data_path']))
        img = img[:,:1280,:]
        pts_img, depth = map_pointcloud_to_image(
            points.copy(), img, 
            copy.deepcopy(cam_intrinsic))
        seq_name = info['cams'][cam_key]['data_path'].split('/')[-3]
        file_name = os.path.split(info['cams'][cam_key]['data_path'])[-1]
        depth_gt = np.concatenate([pts_img, depth[:, None]],
                       axis=1).astype(np.float32).flatten()
        path_to_save = "/mnt/data/DataSet/K-RadarOOC/depth_gt/" +seq_name+file_name+".bin"
        depth_gt.tofile(path_to_save)



if __name__ == '__main__':
    po = Pool(12)
    mmcv.mkdir_or_exist(os.path.join('/mnt/data/K-RadarOOC', 'depth_gt'))
    infos = mmcv.load(info_path_train)['infos']

    for info in infos:
        po.apply_async(func=worker, args=(info, ))
    po.close()
    po.join()
    print("start val")
    po2 = Pool(12)
    infos = mmcv.load(info_path_val)['infos']
    for info in infos:
        po2.apply_async(func=worker, args=(info, ))
    po2.close()
    po2.join()
