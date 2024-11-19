import os
import sys
import yaml
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import chamfer
import mmcv
import open3d as o3d
import numpy as np
from tqdm import tqdm
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
from mmcv.ops.points_in_boxes import (points_in_boxes_all, points_in_boxes_cpu,
                                      points_in_boxes_part)
from scipy.spatial.transform import Rotation
from copy import deepcopy
import multiprocessing


def run_poisson(pcd, depth, n_threads, min_density=None):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, n_threads=8
    )

    # Post-process the mesh
    if min_density:
        vertices_to_remove = densities < np.quantile(densities, min_density)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()

    return mesh, densities

def lidar_to_world_to_lidar(pc,lidar_ego_pose,lidar_ego_pose0):

    pc = LidarPointCloud(pc.T)
    pc.transform(lidar_ego_pose)
    pc.transform(np.linalg.inv(lidar_ego_pose0))

    return pc

def nn_correspondance(verts1, verts2):
    """ for each vertex in verts2 find the nearest vertex in verts1

        Args:
            nx3 np.array's
        Returns:
            ([indices], [distances])

    """
    import open3d as o3d

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances


def preprocess(pcd, config):
    return preprocess_cloud(
        pcd,
        config['max_nn'],
        normals=True
    )

def preprocess_cloud(
    pcd,
    max_nn=20,
    normals=None,
):

    cloud = deepcopy(pcd)
    if normals:
        params = o3d.geometry.KDTreeSearchParamKNN(max_nn)
        cloud.estimate_normals(params)
        cloud.orient_normals_towards_camera_location()

    return cloud

def buffer_to_pointcloud(buffer, compute_normals=False):
    pcd = o3d.geometry.PointCloud()
    for cloud in buffer:
        pcd += cloud
    if compute_normals:
        pcd.estimate_normals()

    return pcd
import glob
import os
LIST_CLS_NAME = [
    'Empty',
    'Static',
    'Sedan',
    'Bus or Truck',
    'Motorcycle',
    'Bicycle',
    'Pedestrian',
    'Pedestrian Group',
    'Bicycle Group',
] 
def get_pc_lidar(path_lidar, type_coord=2,calib_info=None):
    pc_lidar = []

    with open(path_lidar, 'r') as f:
        lines = [line.rstrip('\n') for line in f][13:]
        pc_lidar = [point.split() for point in lines]
        f.close()
    pc_lidar = np.array(pc_lidar, dtype = float).reshape(-1, 9)[:, :4]
    # 0.01: filter out missing values
    # pc_lidar = pc_lidar[np.where(pc_lidar[:, 0] > 0.01)].reshape(-1, 4)
    if type_coord == 1: # Rdr coordinate
        if calib_info is None:
            raise AttributeError('* Exception error (Dataset): Insert calib info!')
        else:
            pc_lidar = np.array(list(map(lambda x: \
                [x[0]+calib_info[0], x[1]+calib_info[1], x[2]+calib_info[2], x[3]],\
                pc_lidar.tolist())))
    return pc_lidar
def process_files(file_path, target_folder):
    """
    Worker function to process a single file.
    """
    pc_lidar = get_pc_lidar(file_path)  # Assuming this is a function you've defined
    # Save the data with a temporary name that includes the original index
    base_name = os.path.basename(file_path)
    temp_name = base_name.replace('.pcd', '.npy')
    np.save(os.path.join(target_folder, temp_name), pc_lidar)

def rename_files(target_folder, n_files):
    """
    Rename files in the target folder from 0.npy to n.npy.
    """
    files = sorted(glob.glob(os.path.join(target_folder, '*.npy')))
    for i, file in enumerate(files[:n_files]):
        new_name = os.path.join(target_folder, f"pc{i}.npy")
        os.rename(file, new_name)

def handle_lidar(source_folder, target_folder):
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)
    files = sorted(glob.glob(source_folder + "/*.pcd"))
    
    num_processes = multiprocessing.cpu_count()

    # Use a process pool to process files concurrently
    with multiprocessing.Pool(num_processes) as pool:
        pool.starmap(process_files, [(file_path, target_folder) for file_path in files])
    
    # After processing, rename files sequentially
    rename_files(target_folder, len(files))
    
def handle_raw_label(source_folder,targer_folder):
    if not os.path.isdir(targer_folder):
        os.makedirs(targer_folder)
    files = sorted(glob.glob(source_folder+"/*.txt"))
    num = -1
    for f in files:
        num+=1
        bbox=[]
        with open(f, 'r') as file:
            firstline = file.readline()  
            spl = firstline.split("*")
            if len(spl) == 3:
                bbox.append(spl[-1][:-1].split(","))
            elif len(spl) > 3:
                print("bad data. CHECK!")
            for line in file:
                bbox.append(line[:-1].split(","))
        bbox_array = []
        tracking_id = []
        obj_category = []
        for b in bbox:
            bbox_array.append([b[3],b[4],b[5],b[7],b[8],b[9],b[6]])
            tracking_id.append(b[1])
            cate_len = len(obj_category)
            for i in range(len(LIST_CLS_NAME)):
                if LIST_CLS_NAME[i].strip() == b[2].strip():
                    obj_category.append(i)
                    break
            if len(obj_category) == cate_len:
                obj_category.append(8)
        
        obj_category = np.array(obj_category).astype(int)
        bbox_array = np.array(bbox_array).astype(float)



        tracking_id = np.array(tracking_id).astype(int)
        tracking_id = tracking_id.astype(str)
        if len(bbox_array) == 0:
            bbox_array= np.empty((0, 7))
            tracking_id = np.empty((0,1))
            obj_category = np.empty((0,1))
            
        bbox_array[:, 2] -= bbox_array[:, 5]
        
        bbox_array[:, 3:6] = bbox_array[:, 3:6] * 2.4

        bbox_array[:, 6:7] = bbox_array[:, 6:7] * np.pi/180 

        np.save(targer_folder+"/bbox_token"+str(num)+".npy",tracking_id)
        np.save(targer_folder+"/bbox"+str(num)+".npy",bbox_array)
        np.save(targer_folder+"/object_category"+str(num)+".npy",obj_category)
def handle_pose(source_files,targer_folder):
    if not os.path.isdir(targer_folder):
        os.makedirs(targer_folder)
    icp_gt = np.load(source_files)
    num = -1
    for i in icp_gt:
        num += 1
        np.save(targer_folder+"/lidar_ego_pose"+str(num)+".npy", i)
def create_mesh_from_map(buffer, depth, n_threads, min_density=None, point_cloud_original= None):

    if point_cloud_original is None:
        pcd = buffer_to_pointcloud(buffer)
    else:
        pcd = point_cloud_original

    return run_poisson(pcd, depth, n_threads, min_density)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parse = ArgumentParser()

    parse.add_argument('--data_path', type=str, default = '/mnt/data/DataSet/K-RadarOOC/')
    parse.add_argument('--config_path', type=str, default='./config.yaml')
    parse.add_argument('--len_sequence', type=int, default = 100)    
    parse.add_argument('--to_mesh', action='store_true', default=False)
    parse.add_argument('--with_semantic', action='store_true', default=True)
    parse.add_argument('--whole_scene_to_mesh', action='store_true', default=False)
    parse.add_argument('--seq2', action='store_true', default=False)
    args=parse.parse_args()

    # load config
    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    voxel_size = config['voxel_size']
    print(voxel_size)
    pc_range = config['pc_range']
    occ_size = config['occ_size']

    # process one clip once
    splits = ['train']
    occ_length = args.len_sequence
    for split in splits:
        
        # currently avaiable seq - 1-4, 6, 8-15, 19 
        # clips = ['19']
        raw_clips = sorted(os.listdir('/mnt/Kradar/K-Radar'))
        for clip in raw_clips:
          path = os.path.join(args.data_path, split, clip)
          pc_path = os.path.join(path,'pc/')

          bbox_path = os.path.join(path,'bbox/')

          seq_file = 0
          print('***Start to process {}***'.format(clip))
          print('***Semantic ***'.format(args.with_semantic))
          path = os.path.join(args.data_path, split, clip)
          pc_path = os.path.join(path,'pc/')
          pc_seman_path = os.path.join(path,'pc/')
          bbox_path = os.path.join(path,'bbox/')
          calib_path = os.path.join(path,'calib/')
          pose_path = os.path.join(path,'pose/')
          save_path = os.path.join(path, 'occupancy_gt/')
          semantic_save_path = os.path.join(path, 'semantic_occupancy_gt/')
          lpc_save_path = os.path.join(path, 'lpc/')
          if not os.path.exists(save_path):
              os.makedirs(save_path)
          if not os.path.exists(bbox_path):
              os.makedirs(bbox_path)
          if not os.path.exists(lpc_save_path):
              os.makedirs(lpc_save_path)
          if not os.path.exists(semantic_save_path):
              os.makedirs(semantic_save_path)
        #   if len(os.listdir(save_path))>0:
        #       print("existing clip, skipped ")
        #       continue
        #   else:
        #       if os.path.exists(pose_path) is False:
        #         print('***No OCC format found, processing raw data ***'.format(args.with_semantic))          
        #         handle_raw_label("/mnt/data/DataSet/K-Radar/KRadar_refined_label_by_UWIPL/"+str(clip)+"/","/mnt/data/DataSet/K-RadarOOC/train/"+str(clip)+"/bbox/")
        #         print("process lidar files")
        #         handle_lidar('/mnt/Kradar/K-Radar/'+str(clip)+'/os2-64','/mnt/data/DataSet/K-RadarOOC/train/'+str(clip)+'/pc')
        #         print("process poses files")
        #         handle_pose('/mnt/Kradar/K-Radar/'+str(clip)+'/results/latest/os2-64_poses.npy','/mnt/data/DataSet/K-RadarOOC/train/'+str(clip)+'/pose')
          handle_raw_label("/mnt/data/DataSet/K-Radar/KRadar_refined_label_by_UWIPL/"+str(clip)+"/","/mnt/data/DataSet/K-RadarOOC/train/"+str(clip)+"/bbox/")

              
          len_sequence = int(len(os.listdir(bbox_path)) /3)
          while seq_file < len_sequence:

            lidar_ego_pose0 = np.load(os.path.join(pose_path, 'lidar_ego_pose0.npy'), allow_pickle=True)
            
            dict_list = []
            file_list = range(seq_file, min(seq_file + occ_length, len_sequence))
            

            for i in tqdm(file_list, desc="*****reading info from dataset*****"):
                if args.with_semantic:
                    pc0 = np.load(os.path.join(pc_seman_path, 'pc{}.npy'.format(i)))
                else:
                    pc0 = np.load(os.path.join(pc_path, 'pc{}.npy'.format(i)))
                boxes = np.load(os.path.join(bbox_path, 'bbox{}.npy'.format(i)))
                object_category = np.load(os.path.join(bbox_path, 'object_category{}.npy'.format(i)))
                boxes_token = np.load(os.path.join(bbox_path, 'bbox_token{}.npy'.format(i)))
                points_in_boxes = points_in_boxes_cpu(torch.from_numpy(pc0[:, :3][np.newaxis, :, :]),
                                              torch.from_numpy(boxes[np.newaxis, :]))
         
                object_points_list = []
                num_boxes = boxes.shape[0]
                if num_boxes>0:
                    for j in range(num_boxes):
                        object_points_mask = points_in_boxes[0][:,j].bool()
                        object_points = pc0[object_points_mask]
                        object_points_list.append(object_points)

                moving_mask = torch.ones_like(points_in_boxes)
                points_in_boxes = torch.sum(points_in_boxes * moving_mask, dim=-1).bool()
                points_mask = ~(points_in_boxes[0])

                ############################# get point mask of the vehicle itself ##########################
                self_range = config['self_range']
                oneself_mask = torch.from_numpy((np.abs(pc0[:, 0]) > self_range[0]) |
                                                (np.abs(pc0[:, 1]) > self_range[1]) |
                                                (np.abs(pc0[:, 2]) > self_range[2]))
                
                ############################# get point mask belonging to the annotated area ################
                ## transfer pc0 to radar coordinates
                ## transfer pc0 to polar
                ## index filter
                pc0_radar = pc0.copy() + np.array([[-2.54, 0.3, 0.7, 0]])
                radar_range_mask = torch.from_numpy((np.linalg.norm(pc0_radar[:,:3], axis=1) < 118))
                radar_fov_mask =  torch.from_numpy(np.abs(np.degrees(np.arctan2(pc0_radar[:,1],pc0_radar[:,0])))<53.5)
                # radar_fov_mask2 =  torch.from_numpy(np.abs(np.degrees(np.arctan2(pc0_radar[:,2],np.linalg.norm(pc0_radar[:,[0,1]],axis=1)))<18.5))  
                # print(radar_fov_mask.shape)
                # print(radar_fov_mask.sum())
                ############################# get static scene segment ##########################
                points_mask = points_mask & oneself_mask
                points_mask = points_mask & radar_range_mask & radar_fov_mask
               
                pc = pc0[points_mask]

                ################## coordinate conversion to the same (first) LiDAR coordinate  ##################
                lidar_ego_pose = np.load(os.path.join(pose_path, 'lidar_ego_pose{}.npy'.format(i)), allow_pickle=True)

                lidar_pc = lidar_to_world_to_lidar(pc.copy(),  lidar_ego_pose.copy(), lidar_ego_pose0)
                
            

                dict = {"object_tokens": boxes_token,
                    "object_points_list": object_points_list,
                    "lidar_pc": lidar_pc.points,
                    "lidar_ego_pose": lidar_ego_pose,
                    "gt_bbox_3d": boxes,
                    "converted_object_category": object_category,
                    "pc_file_name": i}
                dict_list.append(dict)

            

            ################## concatenate all static scene segments  ########################
            lidar_pc_list = [dict['lidar_pc'] for dict in dict_list]
            lidar_pc = np.concatenate(lidar_pc_list, axis=1).T

            #################### save the stiched static lidar point clouds #####################
            for i in range(len(dict_list)):
                lidar_pc_save = np.concatenate(lidar_pc_list[:i+1], axis=1).T
                np.save(lpc_save_path + 'lpc{}.npy'.format(i), lidar_pc_save)


            ################# concatenate all object segments (including non-key frames)  ########################
            object_token_zoo = []
            object_semantic = []
            for dict in dict_list:
                for i,object_token in enumerate(dict['object_tokens']):
                    if object_token not in object_token_zoo:
                        if (dict['object_points_list'][i].shape[0] > 0):
                            object_token_zoo.append(object_token)
                            object_semantic.append(dict['converted_object_category'][i])
                        else:
                            continue

            object_points_dict = {}

            for query_object_token in object_token_zoo:
                object_points_dict[query_object_token] = []
                for dict in dict_list:
                    for i, object_token in enumerate(dict['object_tokens']):
                        if query_object_token == object_token:
                            object_points = dict['object_points_list'][i]
                            if object_points.shape[0] > 0:
                                object_points = object_points[:,:3] - dict['gt_bbox_3d'][i][:3]
                                rots = dict['gt_bbox_3d'][i][6]
                                Rot = Rotation.from_euler('z', -rots, degrees=False)
                                rotated_object_points = Rot.apply(object_points)
                                object_points_dict[query_object_token].append(rotated_object_points)
                        else:
                            continue
                object_points_dict[query_object_token] = np.concatenate(object_points_dict[query_object_token], axis=0)

            object_points_vertice = []
            for key in object_points_dict.keys():
                point_cloud = object_points_dict[key]
                object_points_vertice.append(point_cloud[:,:3])
            # print('object finish')

            if args.whole_scene_to_mesh:
                point_cloud_original = o3d.geometry.PointCloud()
                with_normal2 = o3d.geometry.PointCloud()
                point_cloud_original.points = o3d.utility.Vector3dVector(lidar_pc[:, :3])
                with_normal = preprocess(point_cloud_original, config)
                with_normal2.points = with_normal.points
                with_normal2.normals = with_normal.normals
                mesh, _ = create_mesh_from_map(None, 11, config['n_threads'],
                                            config['min_density'], with_normal2)
                lidar_pc = np.asarray(mesh.vertices, dtype=float)
                lidar_pc = np.concatenate((lidar_pc, np.ones_like(lidar_pc[:,0:1])),axis=1)
                    
            
            for i in tqdm(range(len(dict_list)),desc="*****generating occupancy gt*****"): 
                if i>= len(dict_list):
                    print('finish scene!')
                    break
                dict = dict_list[i]

                ################## convert the static scene to the target coordinate system ##############
                lidar_ego_pose = dict['lidar_ego_pose']
                lidar_pc_i = lidar_to_world_to_lidar(lidar_pc.copy(),
                                                    lidar_ego_pose0.copy(),
                                                    lidar_ego_pose)
                point_cloud = lidar_pc_i.points.T[:,:3]
                if args.with_semantic:        
                    point_cloud_with_semantic = lidar_pc_i.points.T[:,:4]
                    point_cloud_with_semantic[:,3] = 1
                    # 0 for static 

                gt_bbox_3d = dict['gt_bbox_3d']
                gt_bbox_3d[:,6:7] = gt_bbox_3d[:,6:7] 
                locs = gt_bbox_3d[:,0:3]
                dims = gt_bbox_3d[:,3:6]
                rots = gt_bbox_3d[:,6:7]

                ################## bbox placement ##############
                object_points_list = []
                object_semantic_list = []
                for j, object_token in enumerate(dict['object_tokens']):
                    for k, object_token_in_zoo in enumerate(object_token_zoo):
                        if object_token==object_token_in_zoo:
                            points = object_points_vertice[k]
                            Rot = Rotation.from_euler('z', rots[j], degrees=False)
                            rotated_object_points = Rot.apply(points)
                            points = rotated_object_points + locs[j]
                            if points.shape[0] >= 5:
                                points_in_boxes = points_in_boxes_cpu(torch.from_numpy(points[:, :3][np.newaxis, :, :]),
                                                                    torch.from_numpy(gt_bbox_3d[j:j+1][np.newaxis, :]))
                                points = points[points_in_boxes[0,:,0].bool()]

                            object_points_list.append(points)
                            semantics = np.ones_like(points[:,0:1]) * object_semantic[k]
                            object_semantic_list.append(np.concatenate([points[:, :3], semantics], axis=1))

                try: # avoid concatenate an empty array
                    temp = np.concatenate(object_points_list)
                    scene_points = np.concatenate([point_cloud, temp])
                except:
                    scene_points = point_cloud

                if args.with_semantic:
                    try:
                        temp = np.concatenate(object_semantic_list)
                        scene_semantic_points = np.concatenate([point_cloud_with_semantic, temp])
                    except:
                        scene_semantic_points = point_cloud_with_semantic
                
                ################## remain points with a spatial range ##############
                mask = (np.abs(scene_points[:, 0]) < pc_range[3]) & (np.abs(scene_points[:, 1]) < pc_range[4]) \
                    & (scene_points[:, 2] > pc_range[2]) & (scene_points[:, 2] < pc_range[5])
                scene_points = scene_points[mask]

                if args.to_mesh and not args.whole_scene_to_mesh:
                    ################## get mesh via Possion Surface Reconstruction ##############
                    point_cloud_original = o3d.geometry.PointCloud()
                    with_normal2 = o3d.geometry.PointCloud()
                    point_cloud_original.points = o3d.utility.Vector3dVector(scene_points[:, :3])
                    with_normal = preprocess(point_cloud_original, config)
                    with_normal2.points = with_normal.points
                    with_normal2.normals = with_normal.normals
                    mesh, _ = create_mesh_from_map(None, config['depth'], config['n_threads'],
                                                config['min_density'], with_normal2)
                    scene_points = np.asarray(mesh.vertices, dtype=float)


                ################## remain points with a spatial range ##############
                mask = (np.abs(scene_points[:, 0]) < pc_range[3]) & (np.abs(scene_points[:, 1]) < pc_range[4]) \
                    & (scene_points[:, 2] > pc_range[2]) & (scene_points[:, 2] < pc_range[5])
                scene_points = scene_points[mask]

                ################## convert points to voxels ##############
                pcd_np = scene_points
                pcd_np[:, 0] = (pcd_np[:, 0] - pc_range[0]) / voxel_size
                pcd_np[:, 1] = (pcd_np[:, 1] - pc_range[1]) / voxel_size
                pcd_np[:, 2] = (pcd_np[:, 2] - pc_range[2]) / voxel_size
                pcd_np = np.floor(pcd_np).astype(np.int)
                for h in range(len(pcd_np[:, 2])):
                    if pcd_np[h,2] == occ_size[2]:
                        pcd_np[h,2] -= 1
                        print(occ_size[2], " detected")

                voxel = np.zeros(occ_size)
                voxel[pcd_np[:, 0], pcd_np[:, 1], pcd_np[:, 2]] = 1

                ################## convert voxel coordinates to LiDAR system  ##############
                gt_ = voxel
                x = np.linspace(0, gt_.shape[0] - 1, gt_.shape[0])
                y = np.linspace(0, gt_.shape[1] - 1, gt_.shape[1])
                z = np.linspace(0, gt_.shape[2] - 1, gt_.shape[2])
                X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
                vv = np.stack([X, Y, Z], axis=-1)
                fov_voxels = vv[gt_ > 0]
                fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size
                fov_voxels[:, 0] += pc_range[0]
                fov_voxels[:, 1] += pc_range[1]
                fov_voxels[:, 2] += pc_range[2]       # x,y,z in meter

                np.save(save_path + 'occupancy_gt{}.npy'.format(i+seq_file), fov_voxels)
 
                cuda1 = torch.device('cuda:0')

                if args.with_semantic:
                    ################## remain points with a spatial range  ##############
                    mask = (np.abs(scene_semantic_points[:, 0]) < pc_range[3]) & (np.abs(scene_semantic_points[:, 1]) < pc_range[4]) \
                        & (scene_semantic_points[:, 2] > pc_range[2]) & (scene_semantic_points[:, 2] < pc_range[5])
                    scene_semantic_points = scene_semantic_points[mask]

                    ################## Nearest Neighbor to assign semantics ##############
                    dense_voxels = fov_voxels
                    sparse_voxels_semantic = scene_semantic_points

                    x = torch.from_numpy(dense_voxels).cuda(cuda1).unsqueeze(0).float()
                    y = torch.from_numpy(sparse_voxels_semantic[:,:3]).cuda(cuda1).unsqueeze(0).float()
                    d1, d2, idx1, idx2 = chamfer.forward(x,y)
                    indices = idx1[0].cpu().numpy()


                    dense_semantic = sparse_voxels_semantic[:, 3][np.array(indices)]
                    dense_voxels_with_semantic = np.concatenate([fov_voxels, dense_semantic[:, np.newaxis]], axis=1)

                    # to voxel coordinate
                    pcd_np = dense_voxels_with_semantic
                    pcd_np[:, 0] = (pcd_np[:, 0] - pc_range[0]) / voxel_size
                    pcd_np[:, 1] = (pcd_np[:, 1] - pc_range[1]) / voxel_size
                    pcd_np[:, 2] = (pcd_np[:, 2] - pc_range[2]) / voxel_size
                    pcd_np = np.floor(pcd_np).astype(np.int)
                    for h in range(len(pcd_np[:, 2])):
                        if pcd_np[h,2] == occ_size[2]:
                            pcd_np[h,2] -= 1
                            print(occ_size[2], " detected")
                    dense_voxels_with_semantic = pcd_np


                    np.save(semantic_save_path + 'occupancy_gt_with_semantic{}.npy'.format(i+seq_file), dense_voxels_with_semantic)        
               
                    
                i = i + 1
            seq_file += occ_length
