#import open3d as o3d
import time
import trimesh
import mmcv
import numpy as np
import numba as nb

from mmdet.datasets.builder import PIPELINES
import yaml, os
import torch
from scipy import stats
import scipy.io
from scipy.ndimage import zoom
from skimage import transform
import pdb
import torch.nn.functional as F
import copy

def get_arr_in_roi(arr, min_max):
        min_val, max_val = min_max
        idx_min = np.argmin(abs(arr-min_val))
        idx_max = np.argmin(abs(arr-max_val))
        return arr[idx_min:idx_max+1], idx_min, idx_max
        
@PIPELINES.register_module()
class LoadRadarcube(object):
    def __init__(self, radar_cube_path=None,to_float32=True):
        self.to_float32 = to_float32

    def __call__(self, results):
        radar_path = results['radar_path']
        mat_data = scipy.io.loadmat(radar_path)
        ROI= { # each bin is 0.4 [m] (minimum range resolution)
      'z': [-3, 4.6],     # Dim: [m] / [-2.0, 5.6] for Driving corridor / None (erase)
      'y': [-51.2, 50.8], # Dim: [m] / [-6.4, 6.0] for Driving corridor
      'x': [-51.2, 50.8],     # Dim: [m] / [0.0, 71.6] for Driving corridor
            } 
        arr_z_cb = np.arange(-30.7, 32.3, 0.4) ### apply Kradar radar to lidar translation
        arr_y_cb = np.arange(-80.3, 79.7, 0.4)
        arr_x_cb = np.arange(2.54, 102.54, 0.4)
        list_roi_idx_cb = [0, len(arr_z_cb)-1, \
                    0, len(arr_y_cb)-1, 0, len(arr_x_cb)-1]
        idx_attr = 0
        for k, v in ROI.items():
                    if v is not None:
                        min_max = np.array(v).tolist()
                        if k == 'z':
                            arr_roi, idx_min, idx_max = get_arr_in_roi(arr_z_cb, min_max)
                            arr_z_cb = arr_roi
                        if k == 'x':
                            arr_roi, idx_min, idx_max = get_arr_in_roi(arr_x_cb, min_max)
                            arr_x_cb = arr_roi
                        if k == 'y':
                            arr_roi, idx_min, idx_max = get_arr_in_roi(arr_y_cb, min_max)
                            arr_y_cb = arr_roi
                        list_roi_idx_cb[idx_attr*2] = idx_min
                        list_roi_idx_cb[idx_attr*2+1] = idx_max

                    idx_attr += 1
        idx_z_min, idx_z_max, idx_y_min, idx_y_max, idx_x_min, idx_x_max = list_roi_idx_cb
        arr_cube=np.flip(mat_data['arr_zyx'], axis=0)
        arr_cube = arr_cube[idx_z_min:idx_z_max+1,idx_y_min:idx_y_max+1,idx_x_min:idx_x_max+1]
        pad_width = ((0, 0), (0, 0), (134,0 ))  # No padding for the first two axes, 128 padding for the third axis
        arr_cube = np.pad(arr_cube, pad_width, mode='constant', constant_values=0)
        ######################normalization?
        arr_cube = np.maximum(arr_cube, 0.)
        arr_cube /= 1e+13


        results['rdr_cube'] = arr_cube
        return results

@PIPELINES.register_module()
class LoadRadartensor(object):
    def __init__(self, radar_cube_path=None,to_float32=True):
        self.to_float32 = to_float32

    def __call__(self, results):
        radar_path = results['curr']['radar_tensor_path']
        start = time.time()
        mat_data = scipy.io.loadmat(radar_path)
        # print("Time taken: ",time.time()-start)
        arr_cube = mat_data['arrDREA']
        arr_cube = arr_cube[:,:112,:,:] # 51.2m range
        arr_cube /= 1e+13
        results['rdr_tensor'] = arr_cube
        return results

@PIPELINES.register_module()
class LoadRadarPolar(object):
    def __init__(self, to_float32=True):
        self.to_float32 = to_float32

    def __call__(self, results):
        radar_path = results['curr']['radar_polar_path']
        arr_cube = np.load(radar_path)
        results['rdr_cube'] = arr_cube
        return results
    
@PIPELINES.register_module()
class LoadSparseRadar(object):
    def __init__(self, to_float32=True):
        self.to_float32 = to_float32

    def __call__(self, results):
        sparse_radar_path = results['curr']['sparse_radar_path']
        data = np.load(sparse_radar_path)
        # results['power'] = data['power_val']
        # results['range_ind'] = data['range_ind']
        # results['elevation_ind'] = data['elevation_ind']
        # results['azimuth_ind'] = data['azimuth_ind']
        results['sparse_radar'] = data
        return results



@PIPELINES.register_module()
class LoadOccupancy(object):

    def __init__(self, to_float32=True, use_semantic=False, occ_path=None, grid_size=[512, 512, 40], unoccupied=0,
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], gt_resize_ratio=1, cal_visible=False, use_vel=False, simply_label=True):
        self.to_float32 = to_float32
        self.simply_label = simply_label
        self.use_semantic = use_semantic
        self.occ_path = occ_path
        self.cal_visible = cal_visible
        self.grid_size = np.array(grid_size)
        self.unoccupied = unoccupied
        self.pc_range = np.array(pc_range)
        self.voxel_size = (self.pc_range[3:] - self.pc_range[:3]) / self.grid_size
        self.gt_resize_ratio = gt_resize_ratio
        self.use_vel = use_vel

    def __call__(self, results):
        ##############################fov version loaded$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # rel_path = '{0}/semantic_occupancy_gt_fov/occupancy_gt_with_semantic_fov{1}.npy'.format(results['scene_token'], results['lidar_token'])
        abs_occ_path = results['occ_path']
        #  [z y x cls] or [z y x vx vy vz cls]
        pcd = np.load(os.path.join(abs_occ_path))
        pcd_label = pcd[..., -1:]
        pcd_label[pcd_label==0] = 255
        if self.simply_label:
            pcd_label[pcd_label>=3] = 2 #only 0 empty, 1 obstcale, 2 object
        pcd_np_cor = self.voxel2world(pcd[..., [2,1,0]] + 0.5)  # x y z
        untransformed_occ = copy.deepcopy(pcd_np_cor)  # N 4
        # bevdet augmentation
        # pcd_np_cor = (results['bda_mat'] @ torch.from_numpy(pcd_np_cor).unsqueeze(-1).float()).squeeze(-1).numpy()
        # pcd_np_cor = self.world2voxel(pcd_np_cor)

        # make sure the point is in the grid
        pcd_np_cor = np.clip(pcd_np_cor, np.array([0,0,0]), self.grid_size - 1)
        transformed_occ = copy.deepcopy(pcd_np_cor)
        pcd_np = np.concatenate([pcd, pcd_label], axis=-1)

        # velocity
        if self.use_vel:
            pcd_vel = pcd[..., [3,4,5]]  # x y z
            pcd_vel = (results['bda_mat'] @ torch.from_numpy(pcd_vel).unsqueeze(-1).float()).squeeze(-1).numpy()
            pcd_vel = np.concatenate([pcd_np, pcd_vel], axis=-1)  # [x y z cls vx vy vz]
            results['gt_vel'] = pcd_vel

        # 255: noise, 1-16 normal classes, 0 unoccupied
        pcd_np = pcd_np[np.lexsort((pcd_np[:, 0], pcd_np[:, 1], pcd_np[:, 2])), :]
        pcd_np = pcd_np.astype(np.int64)
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.unoccupied
        # print("processed_label shape: ", np.shape(pcd_np))
        processed_label = nb_process_label(processed_label, pcd_np)
        # print("processed_label shape: ", np.shape(processed_label))
        results['gt_occ'] = processed_label

        if self.cal_visible:
            visible_mask = np.zeros(self.grid_size, dtype=np.uint8)
            # camera branch
            if 'img_inputs' in results.keys():
                _, rots, trans, intrins, post_rots, post_trans = results['img_inputs'][:6]
                occ_uvds = self.project_points(torch.Tensor(untransformed_occ), 
                                                rots, trans, intrins, post_rots, post_trans)  # N 6 3
                N, n_cam, _ = occ_uvds.shape
                img_visible_mask = np.zeros((N, n_cam))
                img_h, img_w = results['img_inputs'][0].shape[-2:]
                for cam_idx in range(n_cam):
                    basic_mask = (occ_uvds[:, cam_idx, 0] >= 0) & (occ_uvds[:, cam_idx, 0] < img_w) & \
                                (occ_uvds[:, cam_idx, 1] >= 0) & (occ_uvds[:, cam_idx, 1] < img_h) & \
                                (occ_uvds[:, cam_idx, 2] >= 0)

                    basic_valid_occ = occ_uvds[basic_mask, cam_idx]  # M 3
                    M = basic_valid_occ.shape[0]  # TODO M~=?
                    basic_valid_occ[:, 2] = basic_valid_occ[:, 2] * 10
                    basic_valid_occ = basic_valid_occ.cpu().numpy()
                    basic_valid_occ = basic_valid_occ.astype(np.int16)  # TODO first round then int?
                    depth_canva = np.ones((img_h, img_w), dtype=np.uint16) * 2048
                    nb_valid_mask = np.zeros((M), dtype=np.bool)
                    nb_valid_mask = nb_process_img_points(basic_valid_occ, depth_canva, nb_valid_mask)  # M
                    img_visible_mask[basic_mask, cam_idx] = nb_valid_mask

                img_visible_mask = img_visible_mask.sum(1) > 0  # N  1:occupied  0: free
                img_visible_mask = img_visible_mask.reshape(-1, 1).astype(pcd_label.dtype) 

                img_pcd_np = np.concatenate([transformed_occ, img_visible_mask], axis=-1)
                img_pcd_np = img_pcd_np[np.lexsort((transformed_occ[:, 0], transformed_occ[:, 1], transformed_occ[:, 2])), :]
                img_pcd_np = img_pcd_np.astype(np.int64)
                img_occ_label = np.zeros(self.grid_size, dtype=np.uint8)
                voxel_img = nb_process_label(img_occ_label, img_pcd_np) 
                visible_mask = visible_mask | voxel_img
                results['img_visible_mask'] = voxel_img


            # lidar branch
            if 'points' in results.keys():
                pts = results['points'].tensor.cpu().numpy()[:, :3]
                pts_in_range = ((pts>=self.pc_range[:3]) & (pts<self.pc_range[3:])).sum(1)==3
                pts = pts[pts_in_range]
                pts = (pts - self.pc_range[:3])/self.voxel_size
                pts = np.concatenate([pts, np.ones((pts.shape[0], 1)).astype(pts.dtype)], axis=1) 
                pts = pts[np.lexsort((pts[:, 0], pts[:, 1], pts[:, 2])), :].astype(np.int64)
                pts_occ_label = np.zeros(self.grid_size, dtype=np.uint8)
                voxel_pts = nb_process_label(pts_occ_label, pts)  # W H D 1:occupied 0:free
                visible_mask = visible_mask | voxel_pts
                results['lidar_visible_mask'] = voxel_pts

            results['visible_mask'] = visible_mask

        return results

    def voxel2world(self, voxel):
        """
        voxel: [N, 3]
        """
        return voxel * self.voxel_size[None, :] + self.pc_range[:3][None, :]


    def world2voxel(self, wolrd):
        """
        wolrd: [N, 3]
        """
        return (wolrd - self.pc_range[:3][None, :]) / self.voxel_size[None, :]


    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}'
        return repr_str

    def project_points(self, points, rots, trans, intrins, post_rots, post_trans):
        
        # from lidar to camera
        points = points.reshape(-1, 1, 3)
        points = points - trans.reshape(1, -1, 3)
        inv_rots = rots.inverse().unsqueeze(0)
        points = (inv_rots @ points.unsqueeze(-1))
        
        # from camera to raw pixel
        points = (intrins.unsqueeze(0) @ points).squeeze(-1)
        points_d = points[..., 2:3]
        points_uv = points[..., :2] / points_d
        
        # from raw pixel to transformed pixel
        points_uv = post_rots[:, :2, :2].unsqueeze(0) @ points_uv.unsqueeze(-1)
        points_uv = points_uv.squeeze(-1) + post_trans[..., :2].unsqueeze(0)
        points_uvd = torch.cat((points_uv, points_d), dim=2)
        
        return points_uvd
    
# b1:boolean, u1: uint8, i2: int16, u2: uint16
@nb.jit('b1[:](i2[:,:],u2[:,:],b1[:])', nopython=True, cache=True, parallel=False)
def nb_process_img_points(basic_valid_occ, depth_canva, nb_valid_mask):
    # basic_valid_occ M 3
    # depth_canva H W
    # label_size = M   # for original occ, small: 2w mid: ~8w base: ~30w
    canva_idx = -1 * np.ones_like(depth_canva, dtype=np.int16)
    for i in range(basic_valid_occ.shape[0]):
        occ = basic_valid_occ[i]
        if occ[2] < depth_canva[occ[1], occ[0]]:
            if canva_idx[occ[1], occ[0]] != -1:
                nb_valid_mask[canva_idx[occ[1], occ[0]]] = False

            canva_idx[occ[1], occ[0]] = i
            depth_canva[occ[1], occ[0]] = occ[2]
            nb_valid_mask[i] = True
    return nb_valid_mask

# u1: uint8, u8: uint16, i8: int64
@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label_withvel(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    
    return processed_label


# u1: uint8, u8: uint16, i8: int64
@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label