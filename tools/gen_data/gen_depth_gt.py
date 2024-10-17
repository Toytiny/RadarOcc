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

    # Points live in the point sensor frame. So they need to be
    # transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle
    # frame for the timestamp of the sweep.

    pc = LidarPointCloud(pc.T)
    # pc.rotate(Quaternion(lidar2ego_rotation).rotation_matrix)
    # pc.translate(np.array(lidar2ego_translation))

    # Second step: transform from ego to the global frame.
    # pc.rotate(Quaternion(ego2global_rotation).rotation_matrix)
    # pc.translate(np.array(ego2global_translation))

    # Third step: transform from global into the ego vehicle
    # frame for the timestamp of the image.
    # pc.translate(-np.array(cam_ego2global_translation))
    # pc.rotate(Quaternion(cam_ego2global_rotation).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    t_camera_lidar= np.array([[-0.0079802 , -0.99985409,  0.0151049 ,  0.15099999],
       [ 0.118497  , -0.0159445 , -0.9928264 , -0.461     ],
       [ 0.99292243, -0.0061331 ,  0.1186069 , -0.91500002],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])

    pc.transform(t_camera_lidar)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix
    # + renormalization).
    points = view_points(pc.points[:3, :],
                         cam_intrinsic,
                         normalize=True)

    # Remove points that are either outside or behind the camera.
    # Leave a margin of 1 pixel for aesthetic reasons. Also make
    # sure points are at least 1m in front of the camera to avoid
    # seeing the lidar points on the camera casing for non-keyframes
    # which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.shape[1] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.shape[0] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    return points, coloring


train_ann_file = "/home/xiangyu/SurroundOcc/vod_dict_train.pkl"
val_ann_file = "/home/xiangyu/SurroundOcc/vod_dict_val.pkl"

# data3d_nusc = NuscMVDetData()

cam_keys = [
    'CAM_FRONT'
]


def worker(info):
    lidar_path = info['lidar_lidar']
    points = np.fromfile(lidar_path,
                         dtype=np.float32,
                         count=-1).reshape(-1, 4)[..., :4]


    for i, cam_key in enumerate(cam_keys):

        cam_intrinsic = info['cams'][cam_key]['cam_intrinsic']
        img = mmcv.imread(
            os.path.join(info['cams'][cam_key]['data_path']))
        pts_img, depth = map_pointcloud_to_image(
            points.copy(), img, 
           
            copy.deepcopy(cam_intrinsic))
        
        file_name = os.path.split(info['cams'][cam_key]['data_path'])[-1]
        print(pts_img[:2, :].T.shape, depth[:, None].shape)
        np.concatenate([pts_img[:2, :].T, depth[:, None]],
                       axis=1).astype(np.float32).flatten().tofile(
                           os.path.join('/mnt/data/fangqiang/vod_occ_format/', 'depth_gt',
                                        f'{file_name}.bin'))

if __name__ == '__main__':
    po = Pool(12)
    mmcv.mkdir_or_exist(os.path.join('/mnt/data/fangqiang/vod_occ_format/', 'depth_gt'))
    infos = mmcv.load(train_ann_file)['infos']
    for info in infos:
        # print(info)
        po.apply_async(func=worker, args=(info, ))
    po.close()
    po.join()
    
    po2 = Pool(12)
    infos = mmcv.load(val_ann_file)['infos']
    for info in infos:
        po2.apply_async(func=worker, args=(info, ))
    po2.close()
    po2.join()
