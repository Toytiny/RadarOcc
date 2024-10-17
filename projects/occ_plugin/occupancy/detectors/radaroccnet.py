import torch
import collections 
import torch.nn.functional as F
import torch.nn as nn
from mmdet.models import DETECTORS
from mmcv.runner import auto_fp16, force_fp32
from .bevdepth import BEVDepth
from mmdet3d.models import builder

import numpy as np
import time
import copy

@DETECTORS.register_module()
class RadarOccNet(BEVDepth):
    def __init__(self, 
            loss_cfg=None,
            disable_loss_depth=False,
            empty_idx=0,
            occ_fuser=None,
            occ_encoder_backbone=None,
            occ_encoder_neck=None,
            loss_norm=False,
            **kwargs):
        super().__init__(**kwargs)
                
        self.loss_cfg = loss_cfg
        self.disable_loss_depth = disable_loss_depth
        self.loss_norm = loss_norm
        
        self.record_time = False
        self.time_stats = collections.defaultdict(list)
        self.empty_idx = empty_idx
        self.occ_encoder_backbone = builder.build_backbone(occ_encoder_backbone)
        self.occ_encoder_neck = builder.build_neck(occ_encoder_neck)
        self.occ_fuser = builder.build_fusion_layer(occ_fuser) if occ_fuser is not None else None
        self.pooling_layer = nn.AdaptiveMaxPool3d((128, 128, 10)).cuda()

    # def rdr_cube_encoder(self,rdr_cube):
    #     rdr_cube.shape
            

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
   
        
        # # z_dim, y_dim, x_dim = arr_cube.shape
        # # coords = torch.nonzero(arr_cube, as_tuple=False).cuda() # Shape: [N, 3] where N is the number of non-zero elements

        # # # Extract the non-zero values as features
        # # features = arr_cube[coords[:, 0], coords[:, 1], coords[:, 2]].view(-1, 1)  # Shape: [N, 1]

        # # # Since we have a single batch, we add a column of zeros to represent the batch index
        # # batch_indices = torch.zeros((coords.size(0), 1), dtype=torch.long).cuda()  # Shape: [N, 1]
        # # indices = torch.cat((batch_indices, coords), dim=1)  # Shape: [N, 4]

        # # # Convert to PyTorch tensors
        # # features_tensor = torch.tensor(features, dtype=torch.float32)
        # # indices_tensor = torch.tensor(indices, dtype=torch.int32)

        # # # Define the spatial shape of the 3D radar data and batch size
        # # batch_size = 1  # Only one radar cube
        # # # sparse_tensor = spconv.SparseConvTensor(features_tensor, indices_tensor, spatial_shape, batch_size)
        # radar_data = rdr_cube
        # batch_size, z_dim, y_dim, x_dim = radar_data.size()

        # # Find the indices of the non-zero elements, the result is transposed to [N, 4]
        # coords = torch.nonzero(radar_data, as_tuple=False).type(torch.int32)  # Shape: [N, 4]
        # coords_long = coords.long()
        # # Extract the non-zero values as features
        # features = radar_data[coords_long[:, 0], coords_long[:, 1], coords_long[:, 2], coords_long[:, 3]].view(-1, 1).to(torch.float32)  # Shape: [N, 1]

        # # Since we have a single batch, we can simply use the batch indices already in coords
        # # Note: No need to add a batch column, as the nonzero already returns batch indices.

        # # Convert to PyTorch tensors, ensure indices are int32
        # # Features are already torch.float32 by default.
        # indices_tensor = coords.type(torch.int32)  # Ensure dtype is torch.int32

        radar_data = rdr_cube
        B, z_dim, y_dim, x_dim = radar_data.size() #1,20,256,256

        arr_cube = rdr_cube[0]

        quantile_rate = 0.1
        z_ind, y_ind, x_ind = torch.where(arr_cube > arr_cube.quantile(quantile_rate))

        z_min, z_max =-5, 2.6
        y_min, y_max = -51.2, 50.8
        x_min, x_max = -51.2, 50.8
        power_val = arr_cube[z_ind, y_ind, x_ind].unsqueeze(-1)
        grid_size = 0.4
        z_pc_coord = ((z_min + z_ind * grid_size) - grid_size / 2).unsqueeze(-1)
        y_pc_coord = ((y_min + y_ind * grid_size) - grid_size / 2).unsqueeze(-1)
        x_pc_coord = ((x_min + x_ind * grid_size) - grid_size / 2).unsqueeze(-1)
        sparse_rdr_cube = torch.cat((x_pc_coord, y_pc_coord, z_pc_coord, power_val), dim=-1).to(torch.float32)
        N,C = sparse_rdr_cube.shape
        x_coord, y_coord, z_coord = sparse_rdr_cube[:, 0:1], sparse_rdr_cube[:, 1:2], sparse_rdr_cube[:, 2:3]

        list_batch_indices = []
        for batch_idx in range(B):
            batch_indices = torch.full((N,1), batch_idx, dtype = torch.long)
            list_batch_indices.append(batch_indices)
        sp_indices = torch.cat(list_batch_indices).cuda()

        
        z_ind = torch.ceil((z_coord-z_min) / grid_size).long().cuda()
        y_ind = torch.ceil((y_coord-y_min) / grid_size).long().cuda()
        x_ind = torch.ceil((x_coord-x_min) / grid_size).long().cuda() 

        sp_indices = torch.cat((sp_indices, z_ind, y_ind, x_ind), dim = -1)


        batch_size = 1
        pts_enc_feats = self.pts_middle_encoder(sparse_rdr_cube, sp_indices, batch_size)
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['pts_encoder'].append(t1 - t0)
        
        pts_feats = pts_enc_feats['pts_feats']
        # pts_enc_feats['x']self.pooling_layer(pts_enc_feats['x'])
        return pts_enc_feats['x'], pts_feats

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
            pts_feats=pts_feats,
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
        rdr_cube = kwargs['rdr_cube']

        voxel_feats, img_feats, pts_feats, depth = self.extract_feat(rdr_cube=rdr_cube)

        
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

        voxel_feats, img_feats, pts_feats, depth = self.extract_feat(rdr_cube=rdr_cube)

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
