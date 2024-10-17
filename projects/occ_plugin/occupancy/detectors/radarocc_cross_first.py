import torch
import collections 
import torch.nn.functional as F

from mmdet.models import DETECTORS
from mmcv.runner import auto_fp16, force_fp32
from .bevdepth import BEVDepth
from mmdet3d.models import builder
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.models.utils import build_transformer
import torch.nn as nn


import numpy as np
import time
import copy


# class SphericalToCartesian(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(54 * 19, 512)  # Example dimensions
#         self.fc2 = nn.Linear(512, 128 * 128 * 10)
    
#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # Flatten the input
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         x = x.view(-1, 128, 128, 10)  # Reshape to Cartesian format
#         return x

@DETECTORS.register_module()
class RadarCart(BEVDepth):
    def __init__(self, 
            loss_cfg=None,
            disable_loss_depth=False,
            empty_idx=0,
            self_transformer=None,
            cross_transformer=None,
            positional_encoding=None,
            occ_fuser=None,
            occ_encoder_backbone=None,
            occ_encoder_neck=None,
            loss_norm=False,
            embed_dims = 192,
            **kwargs):
        super().__init__(**kwargs)
                
        self.loss_cfg = loss_cfg
        self.disable_loss_depth = disable_loss_depth
        self.loss_norm = loss_norm
        self.record_time = False
        self.embed_dims = embed_dims
        self.time_stats = collections.defaultdict(list)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        # self.transformer = build_transformer(cross_transformer)
        # self.cart_voxel_emb = nn.Embedding(128*128*14,self.embed_dims)

        self.self_transformer = build_transformer(self_transformer)

    
        self.empty_idx = empty_idx
        self.occ_encoder_backbone = builder.build_backbone(occ_encoder_backbone)
        self.occ_encoder_neck = builder.build_neck(occ_encoder_neck)
        self.occ_fuser = builder.build_fusion_layer(occ_fuser) if occ_fuser is not None else None
        
        self.azimuth_indices, self.elevation_indices, self.range_indices = self.init_param_for_inter()

    # def rdr_cube_encoder(self,rdr_cube):
    #     rdr_cube.shape
    def init_param_for_inter(self):
        x = np.linspace(-51.2, 51.2, 128)
        y = np.linspace(-51.2, 51.2, 128)
        z = np.linspace(-5, 3, 10)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        
        # Convert Cartesian (x, y, z) to Spherical (r, theta, phi)
        r = np.sqrt(xx**2 + yy**2 + zz**2)
        theta = np.arctan2(yy, xx)  # Azimuth
        phi = np.arcsin(zz / r)    # Elevation
        
        # Convert Spherical to indices
        range_idx = r / 0.42 # try from 0.4 to 0.8
        azimuth_idx = (theta * (180 / np.pi) + 54) / 2
        elevation_idx = (phi * (180 / np.pi) + 19) / 2
        
        # Convert numpy arrays to torch tensors
        range_idx = torch.tensor(range_idx, dtype=torch.float32)
        azimuth_idx = torch.tensor(azimuth_idx, dtype=torch.float32)
        elevation_idx = torch.tensor(elevation_idx, dtype=torch.float32)

        return azimuth_idx.cuda(),elevation_idx.cuda(),range_idx.cuda()
    
    def multichannel_trilinear_interpolation(self, polar_data, azimuth_idx, elevation_idx, range_idx):
    # Dimensions

        num_channels, num_range, num_azimuth, num_elevation = polar_data.shape

        # Create a mesh grid and calculate indices and fractions
        azimuth_floor = torch.floor(azimuth_idx).long()
        elevation_floor = torch.floor(elevation_idx).long()
        range_floor = torch.floor(range_idx).long()
        azimuth_frac = azimuth_idx - azimuth_floor
        elevation_frac = elevation_idx - elevation_floor
        range_frac = range_idx - range_floor

        # Compute valid mask
        valid = (azimuth_floor >= 0) & (azimuth_floor < num_azimuth - 1) & (elevation_floor >= 0) & (elevation_floor < num_elevation - 1) & (range_floor >= 0) & (range_floor < num_range - 1)

        # Prepare grids for interpolation
        grid = torch.stack((range_idx, azimuth_idx, elevation_idx), dim=-1)
        grid = grid.unsqueeze(0)  # add batch dimension for grid_sample
        grid[..., 0] = 2 * grid[..., 0] / (num_range - 1) - 1  # normalize to [-1, 1]
        grid[..., 1] = 2 * grid[..., 1] / (num_azimuth - 1) - 1
        grid[..., 2] = 2 * grid[..., 2] / (num_elevation - 1) - 1

        # Perform interpolation
        output = torch.nn.functional.grid_sample(polar_data.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        output = output.squeeze(0)  # remove batch dimension

        # Handle invalid indices
        output[:, ~valid] = 0  # or another specified value

        return output

    import torch

    def cartesian_to_spherical(self,x, y, z):
        r = torch.sqrt(x**2 + y**2 + z**2)
        theta = torch.atan2(y, x)  # azimuth
        phi = torch.asin(z / r)    # elevation
        return r, theta, phi

    def get_reference_points_spherical(self,H=128, W=128, Z=14, bs=1, device='cpu',scale=4, dtype=torch.float,spherical_shape=[64,64,10]):
        # Cartesian coordinates in specified range
        
        grid_size = 0.4 / 2
        x_range = (0, 51.2)
        y_range = (-25.6, 25.6)
        z_range = (-2.4, 3)

        xs = torch.linspace(x_range[0]+grid_size, x_range[1]-grid_size, W, dtype=dtype, device=device)
        ys = torch.linspace(y_range[0]+grid_size, y_range[1]-grid_size, H, dtype=dtype, device=device)
        zs = torch.linspace(z_range[0]+grid_size, z_range[1]-grid_size, Z, dtype=dtype, device=device)

        # Create a meshgrid for 3D coordinates
        xs, ys, zs = torch.meshgrid(xs, ys, zs, indexing='ij')

        # Flatten and stack coordinates
        ref_3d = torch.stack((xs.flatten(), ys.flatten(), zs.flatten()), -1)

        # Convert to spherical coordinates
        rs, thetas, phis = self.cartesian_to_spherical(ref_3d[:, 0], ref_3d[:, 1], ref_3d[:, 2])

        # Scale spherical coordinates to fit into the target grid
        r_res = 0.42 * scale
        theta_res = 1 * scale* (torch.pi / 180)  # Convert degrees to radians
        phi_res = 1 * scale* (torch.pi / 180)    # Convert degrees to radians

        rs_index = (rs / r_res) / spherical_shape[0]
        thetas_index = (((thetas) / theta_res) + (spherical_shape[1]//2)) / spherical_shape[1]
        phis_index = (((phis) / phi_res)  + spherical_shape[2]//2)/ spherical_shape[2]
        ref_3d_spherical = torch.stack([thetas_index,rs_index,phis_index],-1)
        return ref_3d_spherical


    def get_ref_3d(self):
        """Get reference points in 3D.
        Args:
            self.real_h, self.bev_h
        Returns:
            vox_coords (Array): Voxel indices
            ref_3d (Array): 3D reference points
        """
        # not used
        real_h = 102.4
        bev_h = 64
        bev_w = 64
        bev_z = 10
        scene_size = (51.2, 51.2, 8)
        vox_origin = np.array([0, 0, 0])
        voxel_size = real_h / bev_h

        vol_bnds = np.zeros((3,2))
        vol_bnds[:,0] = vox_origin
        vol_bnds[:,1] = vox_origin + np.array(scene_size)

        # Compute the voxels index in lidar cooridnates
        vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
        idx = np.array([range(vol_dim[0]*vol_dim[1]*vol_dim[2])])
        xv, yv, zv = np.meshgrid(range(vol_dim[0]), range(vol_dim[1]), range(vol_dim[2]), indexing='ij')
        vox_coords = np.concatenate([xv.reshape(1,-1), yv.reshape(1,-1), zv.reshape(1,-1), idx], axis=0).astype(int).T

        # Normalize the voxels centroids in lidar cooridnates
        ref_3d = np.concatenate([(xv.reshape(1,-1)+0.5)/bev_h, (yv.reshape(1,-1)+0.5)/bev_w, (zv.reshape(1,-1)+0.5)/bev_z,], axis=0).astype(np.float64).T 
        # print("ref_3d shape {}".format(ref_3d.shape))
        return vox_coords, ref_3d
    

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        
        backbone_feats = self.img_backbone(imgs)
        if self.with_img_neck:
            x = self.img_neck(backbone_feats)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        
        return {'x': x,
                'img_feats': [x.clone()]}
    
    @force_fp32()
    def occ_encoder(self, x):
        x = self.occ_encoder_backbone(x)
        x = self.occ_encoder_neck(x)
        return x
    
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
                
        img_enc_feats = self.image_encoder(img[0])
        x = img_enc_feats['x']
        img_feats = img_enc_feats['img_feats']
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['img_encoder'].append(t1 - t0)

        rots, trans, intrins, post_rots, post_trans, bda = img[1:7]
        
        mlp_input = self.img_view_transformer.get_mlp_input(rots, trans, intrins, post_rots, post_trans, bda)
        geo_inputs = [rots, trans, intrins, post_rots, post_trans, bda, mlp_input]
        
        x, depth = self.img_view_transformer([x] + geo_inputs)

        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['view_transformer'].append(t2 - t1)
        
        return x, depth, img_feats

    def extract_pts_feat(self, rdr_cube):
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        # B, doppler_dim, range_dim, azimuth_dim, elevation_dim = rdr_cube.size() 
        
        power_values = rdr_cube['power_val']
        range_indices, elevation_indices, azimuth_indices = rdr_cube['range_ind'],rdr_cube['elevation_ind'],rdr_cube['azimuth_ind']
        
        dtype = torch.float32
        # rdr_cube = rdr_cube/1e2
        # rdr_cube = torch.mean(rdr_cube, dim=1)
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        batch_size = B = 1
        list_sparse_rdr_cubes = []
        list_sp_indices = []
        for batch_idx in range(B):
            original_k = 800
            n_ranges = 175
            power_val = power_values[batch_idx][:,:original_k*n_ranges]
            elevation_ind = elevation_indices[batch_idx][:original_k*n_ranges]
            azimuth_ind = azimuth_indices[batch_idx][:original_k*n_ranges]
            range_ind = range_indices[batch_idx][:original_k*n_ranges]

            # Change the order to azimuth, range, elevation
            elevation_ind = elevation_ind
            azimuth_ind = azimuth_ind
            range_ind = range_ind
            
            # Step 4: Filter the data to keep only the top k
            k = 250


            # Reshape power values to separate each set of 400
            # power_val = torch.max(power_val, dim=0).values # Doppler max pooling 
            power_val= power_val[2]
            reshaped_power_vals = power_val.view(n_ranges, original_k) 

            # Sort and take top k values from each range bin
            values, indices = torch.sort(reshaped_power_vals, dim=1, descending=True)
            top_k_values = values[:, :k]

            # Get the corresponding indices for top k values
            top_k_indices = indices[:, :k]
            top_k_range_inds = torch.arange(n_ranges).unsqueeze(1).repeat(1, k).cuda()
            top_k_elevation_inds = elevation_ind.view(n_ranges, original_k).gather(1, top_k_indices)
            top_k_azimuth_inds = azimuth_ind.view(n_ranges, original_k).gather(1, top_k_indices)

            # Flattening the results if necessary
            final_range_inds = top_k_range_inds.flatten()
            final_elevation_inds = top_k_elevation_inds.flatten()
            final_azimuth_inds = top_k_azimuth_inds.flatten()
            #37,107,256
            power_val = rdr_cube['power_val'][batch_idx]
            sparse_rdr_cube =  torch.swapaxes(power_val,0,1)[top_k_indices.flatten(),:]
            list_sparse_rdr_cubes.append(sparse_rdr_cube)


            N, C = sparse_rdr_cube.shape
  
            batch_indices = torch.full((N, 1), batch_idx, dtype=torch.long).cuda()
            sp_indices = torch.cat((batch_indices, final_elevation_inds.unsqueeze(-1),final_range_inds.unsqueeze(-1),final_azimuth_inds.unsqueeze(-1)), dim=-1)
            list_sp_indices.append(sp_indices)

        sparse_rdr_cube_all_batches = torch.cat(list_sparse_rdr_cubes, dim=0)
        sp_indices_all_batches = torch.cat(list_sp_indices, dim=0).cuda()
        pts_enc_feats = self.pts_middle_encoder(sparse_rdr_cube_all_batches, sp_indices_all_batches, batch_size)

        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['pts_encoder'].append(t1 - t0)
        
        pts_feats = pts_enc_feats['pts_feats']
        spherical_feat = pts_enc_feats['x']
        h,w,z = [128,128,14]
        spherical_pos_self= self.positional_encoding(torch.zeros((batch_size, h,w,z)).to(dtype).cuda()).to(dtype) 
        # 44,27
        vox_feats_flatten = spherical_feat.flatten(2) 
        vox_coords, ref_3d = self.get_ref_3d()
        # start.record()

        vox_feats_diff = self.self_transformer.diffuse_vox_features(
            vox_feats_flatten,
            h,
            w,
            ref_3d=ref_3d,
            spatial_shapes = [h,w,z],
            vox_coords=None,
            bev_pos=spherical_pos_self,
            prev_bev=None,
        )
        # vox_feats_diff= vox_feats_diff.permute(0,2,1)
        voxel_feat = vox_feats_diff.reshape((1,self.embed_dims,h,w,z))


        # voxel_queries = self.cart_voxel_emb.weight.to(dtype)
        # bev_pos_self_attn = self.positional_encoding(torch.zeros((batch_size, 128,128,20)).to(dtype).cuda()).to(dtype) 
        # spherical_pos= self.positional_encoding(torch.zeros((batch_size, 64,64,10)).to(dtype).cuda()).to(dtype) 
        # voxel_queries = voxel_queries.reshape(batch_size, self.embed_dims,-1) 
        # sph_ref_3d = self.get_reference_points_spherical(device=spherical_feat.device,spherical_shape=[64,64,10])

        # vox_feats_diff = self.transformer.get_cart_features(
        #     voxel_queries,
        #     voxel_feat,
        #     128,
        #     128,
        #     ref_3d = sph_ref_3d,
        #     sph_ref_3d = sph_ref_3d,
        #     spatial_shapes = [64,64,10],
        #     cart_spatial_shape = [128,128,20],
        #     vox_coords=None,
        #     bev_pos=bev_pos_self_attn,
        #     spherical_pos=spherical_pos,
        #     prev_bev=None,
        # )
        # voxel_feat = vox_feats_diff.reshape((1,192,128,128,20))
        # voxel_feat =  voxel_feat + inter_feat



        return voxel_feat, pts_feats

    def extract_feat(self, rdr_cube):
        """Extract features from images and points."""
        img_voxel_feats = None
        pts_voxel_feats, pts_feats = None, None
        depth, img_feats = None, None

        pts_voxel_feats, pts_feats = self.extract_pts_feat(rdr_cube)

        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()

        if self.occ_fuser is not None:
            voxel_feats = self.occ_fuser(img_voxel_feats, pts_voxel_feats)
        else:
            assert (img_voxel_feats is None) or (pts_voxel_feats is None)
            voxel_feats = img_voxel_feats if pts_voxel_feats is None else pts_voxel_feats

        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['occ_fuser'].append(t1 - t0)

        voxel_feats_enc = self.occ_encoder(voxel_feats)
        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]

        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['occ_encoder'].append(t2 - t1)

        return (voxel_feats_enc, img_feats, pts_feats, depth)
    
    @force_fp32(apply_to=('voxel_feats'))
    def forward_pts_train(
            self,
            voxel_feats,
            gt_occ=None,
            points_occ=None,
            img_metas=None,
            transform=None,
            img_feats=None,
            pts_feats=None,
            visible_mask=None,
        ):
        
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        
        outs = self.pts_bbox_head(
            voxel_feats=voxel_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            pts_feats=None,
            transform=transform,
        )
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['occ_head'].append(t1 - t0)
        
        losses = self.pts_bbox_head.loss(
            output_voxels=outs['output_voxels'],
            output_voxels_fine=outs['output_voxels_fine'],
            output_coords_fine=outs['output_coords_fine'],
            target_voxels=gt_occ,
            target_points=points_occ,
            img_metas=img_metas,
            visible_mask=visible_mask,
        )
        
        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['loss_occ'].append(t2 - t1)
        
        return losses
    
    def forward_train(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            gt_occ=None,
            points_occ=None,
            visible_mask=None,
            **kwargs,
        ):

        sparse_radar = kwargs['sparse_radar']

        voxel_feats, img_feats, pts_feats, depth = self.extract_feat(rdr_cube=sparse_radar)

        
        # training losses
        losses = dict()
        
        if self.record_time:        
            torch.cuda.synchronize()
            t0 = time.time()
        
        if not self.disable_loss_depth and depth is not None:
            losses['loss_depth'] = self.img_view_transformer.get_depth_loss(img_inputs[-2], depth)
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['loss_depth'].append(t1 - t0)
        
        transform = img_inputs[1:8] if img_inputs is not None else None
        losses_occupancy = self.forward_pts_train(voxel_feats, gt_occ,
                        points_occ, img_metas, img_feats=img_feats, pts_feats=pts_feats, transform=transform, 
                        visible_mask=visible_mask)
        losses.update(losses_occupancy)
        if self.loss_norm:
            pass
            l1_deatch = 0
            for loss_key in losses.keys():

                if loss_key.startswith('loss'):
                    losses[loss_key] = losses[loss_key] / (losses[loss_key].detach() + 1e-9)
            # for loss_key in losses.keys():
            #     if loss_key.startswith('loss_voxel_sem') is False:
            #         losses[loss_key] = losses[loss_key] / (losses[loss_key].detach() + 1e-9) * l1_deatch
            #     # L = L1 + L2 / L2.detach() * L1.detach() + L3 / L3.detach() * L1.detach()
        def logging_latencies():
            # logging latencies
            avg_time = {key: sum(val) / len(val) for key, val in self.time_stats.items()}
            sum_time = sum(list(avg_time.values()))
            out_res = ''
            for key, val in avg_time.items():
                out_res += '{}: {:.4f}, {:.1f}, '.format(key, val, val / sum_time)
            
            print(out_res)
        
        if self.record_time:
            logging_latencies()
        
        return losses
        
    def forward_test(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            gt_occ=None,
            visible_mask=None,
            **kwargs,
        ):
        return self.simple_test(img_metas, img_inputs, points, gt_occ=gt_occ, visible_mask=visible_mask, **kwargs)
    
    def simple_test(self, img_metas, img=None, points=None, rescale=False, points_occ=None, 
            gt_occ=None, visible_mask=None, **kwargs):
        rdr_cube = kwargs['sparse_radar']

        voxel_feats, img_feats, pts_feats, depth = self.extract_feat(rdr_cube=rdr_cube)

        transform = img[1:8] if img is not None else None
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            pts_feats=pts_feats,
            transform=transform,
        )
        pred_c = output['output_voxels'][0]
        target_voxels = gt_occ
        B, C, H, W, D = pred_c.shape
        ratio = target_voxels.shape[2] // H
        if ratio != 1:
            target_voxels = target_voxels.reshape(B, H, ratio, W, ratio, D, ratio).permute(0,1,3,5,2,4,6).reshape(B, H, W, D, ratio**3)
            empty_mask = target_voxels.sum(-1) == self.empty_idx
            target_voxels = target_voxels.to(torch.int64)
            occ_space = target_voxels[~empty_mask]
            occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1
            target_voxels[~empty_mask] = occ_space
            target_voxels = torch.mode(target_voxels, dim=-1)[0]
            target_voxels[target_voxels<0] = 255
            target_voxels = target_voxels.long()
        

        
        SC_metric, SC_metric_range1,SC_metric_range2 = self.evaluation_semantic(pred_c, target_voxels, eval_type='SC', visible_mask=visible_mask)
        SSC_metric, SSC_metric_range1, SSC_metric_range2 = self.evaluation_semantic(pred_c, target_voxels, eval_type='SSC', visible_mask=visible_mask)

        pred_f = None
        SSC_metric_fine = None
        if output['output_voxels_fine'] is not None:
            if output['output_coords_fine'] is not None:
                fine_pred = output['output_voxels_fine'][0]  # N ncls
                fine_coord = output['output_coords_fine'][0]  # 3 N
                pred_f = self.empty_idx * torch.ones_like(gt_occ)[:, None].repeat(1, fine_pred.shape[1], 1, 1, 1).float()
                pred_f[:, :, fine_coord[0], fine_coord[1], fine_coord[2]] = fine_pred.permute(1, 0)[None]
            else:
                pred_f = output['output_voxels_fine'][0]
            SC_metric, _ = self.evaluation_semantic(pred_f, gt_occ, eval_type='SC', visible_mask=visible_mask)
            SSC_metric_fine, SSC_occ_metric_fine = self.evaluation_semantic(pred_f, gt_occ, eval_type='SSC', visible_mask=visible_mask)

        test_output = {
            'SC_metric': SC_metric,
            'SC_metric_1': SC_metric_range1,
            'SC_metric_2': SC_metric_range2,
            'SSC_metric': SSC_metric,
            'SSC_metric_1': SSC_metric_range1,
            'SSC_metric_2': SSC_metric_range2,
            'pred_c': pred_c,
            'pred_f': pred_f,
        }

        if SSC_metric_fine is not None:
            test_output['SSC_metric_fine'] = SSC_metric_fine

        return test_output


    def evaluation_semantic(self, pred, gt, eval_type, visible_mask=None):
        _, H, W, D = gt.shape
        # before_inter = torch.argmax(pred[0], dim=0).cpu().numpy()

        # pred = F.interpolate(pred, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
        pred = torch.argmax(pred[0], dim=0).cpu().numpy()
        gt = gt[0].cpu().numpy()
        gt = gt.astype(np.int)
        

        # ignore noise
        noise_mask = gt != 255

        if eval_type == 'SC':
            # 0 1 split

            gt[gt != self.empty_idx] = 1
            pred[pred != self.empty_idx] = 1

            range2=  fast_hist(pred[:32,48:80,:], gt[:32,48:80,:], max_label=2)
            range1 = fast_hist(pred[:64,32:96,:], gt[:64,32:96,:], max_label=2)


            return fast_hist(pred[noise_mask], gt[noise_mask], max_label=2), range1,range2


        if eval_type == 'SSC':
            hist_occ = None
            if visible_mask is not None:
                visible_mask = visible_mask[0].cpu().numpy()
                mask = noise_mask & (visible_mask!=0)
                hist_occ = fast_hist(pred[mask], gt[mask], max_label=3)

            range2=  fast_hist(pred[:32,48:80,:], gt[:32,48:80,:], max_label=3)
            range1 = fast_hist(pred[:64,32:96,:], gt[:64,32:96,:], max_label=3)
            hist = fast_hist(pred[noise_mask], gt[noise_mask], max_label=3)
            return hist, range1, range2
    
    
    def forward_dummy(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            points_occ=None,
            **kwargs,
        ):
        sparse_radar = kwargs['sparse_radar']
        voxel_feats, img_feats, pts_feats, depth = self.extract_feat(rdr_cube=sparse_radar)

        transform = img_inputs[1:8] if img_inputs is not None else None
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            pts_feats=pts_feats,
            transform=transform,
        )
        
        return output
    
    
def fast_hist(pred, label, max_label=18):
    pred = copy.deepcopy(pred.flatten())
    label = copy.deepcopy(label.flatten())
    bin_count = np.bincount(max_label * label.astype(int) + pred, minlength=max_label ** 2)
    return bin_count[:max_label ** 2].reshape(max_label, max_label)
