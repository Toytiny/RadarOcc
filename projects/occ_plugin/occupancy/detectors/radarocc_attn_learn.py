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

@DETECTORS.register_module()
class RadarOccAttn_learn(BEVDepth):
    def __init__(self, 
            loss_cfg=None,
            disable_loss_depth=False,
            empty_idx=0,
            self_transformer=None,
            positional_encoding=None,
            occ_fuser=None,
            occ_encoder_backbone=None,
            occ_encoder_neck=None,
            loss_norm=False,
            cfar=None,
            **kwargs):
        super().__init__(**kwargs)
                
        self.loss_cfg = loss_cfg
        self.disable_loss_depth = disable_loss_depth
        self.loss_norm = loss_norm
        self.cfar = builder.build_neck(cfar)
        
        self.record_time = False
        self.time_stats = collections.defaultdict(list)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.self_transformer = build_transformer(self_transformer)
        self.empty_idx = empty_idx
        self.occ_encoder_backbone = builder.build_backbone(occ_encoder_backbone)
        self.occ_encoder_neck = builder.build_neck(occ_encoder_neck)
        self.occ_fuser = builder.build_fusion_layer(occ_fuser) if occ_fuser is not None else None
        self.embed_dims = 192

    # def rdr_cube_encoder(self,rdr_cube):
    #     rdr_cube.shape
    
    def get_ref_3d(self):
        """Get reference points in 3D.
        Args:
            self.real_h, self.bev_h
        Returns:
            vox_coords (Array): Voxel indices
            ref_3d (Array): 3D reference points
        """
        real_h = 102.4
        bev_h = 128
        bev_w = 128
        bev_z = 10
        scene_size = (102.4, 102.4, 8)
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

        radar_data = rdr_cube
        B, z_dim, y_dim, x_dim = radar_data.size() #1,20,256,256

        arr_cube = rdr_cube

        z_min, z_max =-5, 2.6
        y_min, y_max = -51.2, 50.8
        x_min, x_max = -51.2, 50.8

        cube_sigmoid, z_ind, y_ind, x_ind = self.cfar(arr_cube)


        
        assert(z_ind.shape[0]%B==0)
        N = z_ind.shape[0] / B
        
        list_batch_indices = []
        for batch_idx in range(B):
            batch_indices = torch.full((int(N),1), batch_idx, dtype = torch.long)
            list_batch_indices.append(batch_indices)
        b_ind = torch.cat(list_batch_indices).squeeze(1)

        power_val = arr_cube[b_ind,z_ind, y_ind, x_ind].unsqueeze(-1)
        grid_size = 0.4
        z_pc_coord = ((z_min + z_ind * grid_size) - grid_size / 2).unsqueeze(-1)
        y_pc_coord = ((y_min + y_ind * grid_size) - grid_size / 2).unsqueeze(-1)
        x_pc_coord = ((x_min + x_ind * grid_size) - grid_size / 2).unsqueeze(-1)
        sparse_rdr_cube = torch.cat((x_pc_coord, y_pc_coord, z_pc_coord, power_val), dim=-1).to(torch.float32)

        sp_indices = torch.cat(list_batch_indices).cuda()
        dtype = torch.float32
        
        z_ind = z_ind.unsqueeze(-1).long().cuda()
        y_ind = y_ind.unsqueeze(-1).long().cuda()
        x_ind = x_ind.unsqueeze(-1).long().cuda() 

        sp_indices = torch.cat((sp_indices, z_ind, y_ind, x_ind), dim = -1)                


        batch_size = B
        pts_enc_feats = self.pts_middle_encoder(sparse_rdr_cube, sp_indices, batch_size)
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['pts_encoder'].append(t1 - t0)
        
        pts_feats = pts_enc_feats['pts_feats']
        # print("pts_enc_feats['x'] shape is {}".format(pts_enc_feats['x'].shape))
        voxel_feat = pts_enc_feats['x']

        bev_pos_self_attn = self.positional_encoding(torch.zeros((batch_size, 128, 128,10)).to(dtype).cuda()).to(dtype) 
        vox_feats_flatten = pts_enc_feats['x'].reshape(batch_size, self.embed_dims,-1) #check? b,192,
        print("voxel_feat shape {}".format(pts_enc_feats['x'].shape))
        print("voxel_Feats_flatten shape  {}".format(vox_feats_flatten.shape))

        vox_coords, ref_3d = self.get_ref_3d()

        vox_feats_diff = self.self_transformer.diffuse_vox_features(
            vox_feats_flatten,
            128,
            128,
            ref_3d=ref_3d,
            vox_coords=vox_coords,
            bev_pos=bev_pos_self_attn,
            prev_bev=None,
            batch_size = batch_size,
        )
        voxel_feat = vox_feats_diff.reshape((batch_size,192,128,128,10))

        # Reshape voxel_feat back to the desired output format

        return voxel_feat, pts_feats, cube_sigmoid

    def extract_feat(self, rdr_cube):
        """Extract features from images and points."""
        img_voxel_feats = None
        pts_voxel_feats, pts_feats = None, None
        depth, img_feats = None, None

        pts_voxel_feats, pts_feats, cube_sigmoid = self.extract_pts_feat(rdr_cube)

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

        return (voxel_feats_enc, img_feats, pts_feats, depth, cube_sigmoid)
    
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
            sigmoid_cube=None
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
            sigmoid_cube = sigmoid_cube,
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
        rdr_cube = kwargs['rdr_cube']

        voxel_feats, img_feats, pts_feats, depth, cube_sigmoid = self.extract_feat(rdr_cube=rdr_cube)

        
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
            for loss_key in losses.keys():
                if loss_key.startswith('loss'):
                    losses[loss_key] = losses[loss_key] / (losses[loss_key].detach() + 1e-9)

        def logging_latencies():
            # logging latencies
            avg_time = {key: sum(val) / len(val) for key, val in self.time_stats.items()}
            sum_time = sum(list(avg_time.values()))
            out_res = ''
            for key, val in avg_time.items():
                out_res += '{}: {:.4f}, {:.1f}, '.format(key, val, val / sum_time)
            
            x(out_res)
        
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
        rdr_cube = kwargs['rdr_cube']

        voxel_feats, img_feats, pts_feats, depth, cube_sigmoid = self.extract_feat(rdr_cube=rdr_cube)

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
        SC_metric, _ = self.evaluation_semantic(pred_c, gt_occ, eval_type='SC', visible_mask=visible_mask)
        SSC_metric, SSC_occ_metric = self.evaluation_semantic(pred_c, gt_occ, eval_type='SSC', visible_mask=visible_mask)

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
            'SSC_metric': SSC_metric,
            'pred_c': pred_c,
            'pred_f': pred_f,
        }

        if SSC_metric_fine is not None:
            test_output['SSC_metric_fine'] = SSC_metric_fine

        return test_output


    def evaluation_semantic(self, pred, gt, eval_type, visible_mask=None):
        _, H, W, D = gt.shape
        before_inter = torch.argmax(pred[0], dim=0).cpu().numpy()
        # print("before inter pred: ",np.unique(before_inter,return_counts=True))

        pred = F.interpolate(pred, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
        pred = torch.argmax(pred[0], dim=0).cpu().numpy()
        gt = gt[0].cpu().numpy()
        gt = gt.astype(np.int)

        # ignore noise
        noise_mask = gt != 255

        if eval_type == 'SC':
            # 0 1 split
            # print("gt: ",np.unique(gt,return_counts=True),"    pred: ",np.unique(pred,return_counts=True))

            gt[gt != self.empty_idx] = 1
            pred[pred != self.empty_idx] = 1

            return fast_hist(pred[noise_mask], gt[noise_mask], max_label=2), None


        if eval_type == 'SSC':
            hist_occ = None
            if visible_mask is not None:
                visible_mask = visible_mask[0].cpu().numpy()
                mask = noise_mask & (visible_mask!=0)
                hist_occ = fast_hist(pred[mask], gt[mask], max_label=17)

            hist = fast_hist(pred[noise_mask], gt[noise_mask], max_label=17)
            return hist, hist_occ
    
    def forward_dummy(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            points_occ=None,
            **kwargs,
        ):
        rdr_cube = kwargs['rdr_cube']

        voxel_feats, img_feats, pts_feats, depth,cube_sigmoid = self.extract_feat(rdr_cube=rdr_cube)

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
