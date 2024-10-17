import math
from functools import partial
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import spconv.pytorch as spconv
from spconv.pytorch import functional as Fsp

from mmdet3d.models.builder import MIDDLE_ENCODERS

import copy

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_cfg=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        build_norm_layer(norm_cfg, out_channels)[1],
        nn.ReLU(inplace=True),
    )

    return m



class SparseBasicBlock(spconv.SparseModule):

    def __init__(self, inplanes, planes, stride=1, kernel_size=3, norm_cfg=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        padding = (kernel_size-1)//2

        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, indice_key=indice_key),
            build_norm_layer(norm_cfg, planes)[1],
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, indice_key=indice_key),
            build_norm_layer(norm_cfg, planes)[1],
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.net(x)
        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))

        return out



@MIDDLE_ENCODERS.register_module()
class SparseLiDAREnc4x(nn.Module):
    def __init__(self, input_channel, norm_cfg, base_channel, out_channel, 
                sparse_shape_xyz, **kwargs):
        super().__init__()

        block = post_act_block
        self.sparse_shape_xyz = sparse_shape_xyz

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channel, base_channel, 3),
            nn.GroupNorm(16, base_channel),
            nn.ReLU(inplace=True))

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(base_channel, base_channel, norm_cfg=norm_cfg, indice_key='res1'),
            SparseBasicBlock(base_channel, base_channel, norm_cfg=norm_cfg, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            block(base_channel, base_channel*2, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, indice_key='res2'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            block(base_channel*2, base_channel*4, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(base_channel*4, base_channel*4, norm_cfg=norm_cfg, indice_key='res3'),
            SparseBasicBlock(base_channel*4, base_channel*4, norm_cfg=norm_cfg, indice_key='res3'),
        )

        self.conv_out = spconv.SparseSequential(
            spconv.SubMConv3d(base_channel*4, out_channel, 3),
            nn.GroupNorm(16, out_channel),
            nn.ReLU(inplace=True))



    def forward(self, voxel_features, coors, batch_size):
        # spconv encoding
        coors = coors.int()  # zyx 与 points的输入(xyz)相反
        # FIXME bs=1 hardcode
        input_sp_tensor = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape_xyz[::-1], batch_size)
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        
        x = self.conv_out(x_conv3)

        return {'x': x.dense().permute(0,1,4,3,2), # B, C, W, H, D 
                'pts_feats': [x]}





@MIDDLE_ENCODERS.register_module()
class SparseLiDAREnc8x(nn.Module):
    def __init__(self, input_channel, norm_cfg, base_channel, out_channel, 
                sparse_shape_xyz, **kwargs):
        super().__init__()

        block = post_act_block
        self.sparse_shape_xyz = sparse_shape_xyz

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channel, base_channel, 3),
            nn.GroupNorm(16, base_channel),
            nn.ReLU(inplace=True))

        self.conv1 = spconv.SparseSequential(
            block(base_channel, base_channel*2, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv1', conv_type='spconv'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, indice_key='res1'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            block(base_channel*2, base_channel*4, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(base_channel*4, base_channel*4, norm_cfg=norm_cfg, indice_key='res2'),
            SparseBasicBlock(base_channel*4, base_channel*4, norm_cfg=norm_cfg, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            block(base_channel*4, base_channel*8, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(base_channel*8, base_channel*8, norm_cfg=norm_cfg, indice_key='res3'),
            SparseBasicBlock(base_channel*8, base_channel*8, norm_cfg=norm_cfg, indice_key='res3'),
        )

        self.conv_out = spconv.SparseSequential(
            spconv.SubMConv3d(base_channel*8, out_channel, 3),
            nn.GroupNorm(16, out_channel),
            nn.ReLU(inplace=True))



    def forward(self, voxel_features, coors, batch_size):
        # spconv encoding
        coors = coors.int()  # zyx 与 points的输入(xyz)相反
        # FIXME bs=1 hardcode
        # print(coors)

        input_sp_tensor = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape_xyz[::-1], batch_size)
        # print(voxel_features)
        x = self.conv_input(input_sp_tensor)
        # print(x.dense().permute(0,1,4,3,2).shape)
        x_conv1 = self.conv1(x)
        # print(x.dense().permute(0,1,4,3,2).shape)
        x_conv2 = self.conv2(x_conv1)
        # print(x.dense().permute(0,1,4,3,2).shape)
        x_conv3 = self.conv3(x_conv2)
        # print(x.dense().permute(0,1,4,3,2).shape)
        x = self.conv_out(x_conv3)
        # print(x.dense().permute(0,1,4,3,2).shape)
        # print(x)
        return {'x': x.dense().permute(0,1,4,3,2), # B, C, W, H, D 
                'pts_feats': [x]}


@MIDDLE_ENCODERS.register_module()
class SparseRADAREnc8x(nn.Module):
    def __init__(self, input_channel, norm_cfg, base_channel, out_channel, 
                sparse_shape_xyz, **kwargs):
        super().__init__()

        block = post_act_block
        self.sparse_shape_xyz = sparse_shape_xyz
        self.conv_radar_input = spconv.SparseSequential(
            spconv.SubMConv3d(1, 4, 3),
            nn.GroupNorm(2, 4),
            nn.ReLU(inplace=True))

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channel, base_channel, 3),
            nn.GroupNorm(16, base_channel),
            nn.ReLU(inplace=True))

        self.conv1 = spconv.SparseSequential(
            block(base_channel, base_channel*2, 3, norm_cfg=norm_cfg, stride=1, padding=1, indice_key='spconv1', conv_type='spconv'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, indice_key='res1'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            block(base_channel*2, base_channel*4, 3, norm_cfg=norm_cfg, stride=1, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(base_channel*4, base_channel*4, norm_cfg=norm_cfg, indice_key='res2'),
            SparseBasicBlock(base_channel*4, base_channel*4, norm_cfg=norm_cfg, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            block(base_channel*4, base_channel*8, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(base_channel*8, base_channel*8, norm_cfg=norm_cfg, indice_key='res3'),
            SparseBasicBlock(base_channel*8, base_channel*8, norm_cfg=norm_cfg, indice_key='res3'),
        )

        self.conv_out = spconv.SparseSequential(
            spconv.SubMConv3d(base_channel*8, out_channel, 3),
            nn.GroupNorm(16, out_channel),
            nn.ReLU(inplace=True))



    def forward(self, voxel_features, coors, batch_size):
        # spconv encoding
        coors = coors.int()  # zyx 与 points的输入(xyz)相反
        # FIXME bs=1 hardcode
        # z,y,x [150, 400, 250] -> 

        input_sp_tensor = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape_xyz[::-1], batch_size)
     
        x = self.conv_radar_input(input_sp_tensor)
        x = self.conv_input(x)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x = self.conv_out(x_conv3)
        return {'x': x.dense().permute(0,1,4,3,2), # B, C, W, H, D 
                'pts_feats': [x]}





@MIDDLE_ENCODERS.register_module()
class SparseRaDAREnc2x(nn.Module):
    def __init__(self, input_channel, norm_cfg, base_channel, out_channel, 
                sparse_shape_xyz, **kwargs):
        super().__init__()

        block = post_act_block
        self.sparse_shape_xyz = sparse_shape_xyz

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channel, base_channel, 3),
            nn.GroupNorm(16, base_channel),
            nn.ReLU(inplace=True))

        self.conv1 = spconv.SparseSequential(
            block(base_channel, base_channel*2, 3, norm_cfg=norm_cfg, stride=1, padding=1, indice_key='spconv1', conv_type='spconv'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, indice_key='res1'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            block(base_channel*2, base_channel*4, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(base_channel*4, base_channel*4, norm_cfg=norm_cfg, indice_key='res2'),
            SparseBasicBlock(base_channel*4, base_channel*4, norm_cfg=norm_cfg, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            block(base_channel*4, base_channel*8, 3, norm_cfg=norm_cfg, stride=1, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(base_channel*8, base_channel*8, norm_cfg=norm_cfg, indice_key='res3'),
            SparseBasicBlock(base_channel*8, base_channel*8, norm_cfg=norm_cfg, indice_key='res3'),
        )

        self.conv_out = spconv.SparseSequential(
            spconv.SubMConv3d(base_channel*8, out_channel, 3),
            nn.GroupNorm(16, out_channel),
            nn.ReLU(inplace=True))



    def forward(self, voxel_features, coors, batch_size):
        # spconv encoding
        coors = coors.int()  # zyx 与 points的输入(xyz)相反
        input_sp_tensor = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape_xyz[::-1], batch_size)

        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)

        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x = self.conv_out(x_conv3)


        return {'x': x.dense().permute(0,1,4,3,2), # B, C, W, H, D 
                'pts_feats': [x]}


@MIDDLE_ENCODERS.register_module()
class SparsePolarRaDAREnc2x(nn.Module):
    def __init__(self, input_channel, norm_cfg, base_channel, out_channel, 
                sparse_shape_xyz, **kwargs):
        super().__init__()

        block = post_act_block
        self.sparse_shape_xyz = sparse_shape_xyz

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channel, base_channel, 3),
            nn.GroupNorm(16, base_channel),
            nn.ReLU(inplace=True))

        self.conv1 = spconv.SparseSequential(
            block(base_channel, base_channel*2, 3, norm_cfg=norm_cfg, stride=1, padding=1, indice_key='spconv1', conv_type='spconv'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, indice_key='res1'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            block(base_channel*2, base_channel*4, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(base_channel*4, base_channel*4, norm_cfg=norm_cfg, indice_key='res2'),
            SparseBasicBlock(base_channel*4, base_channel*4, norm_cfg=norm_cfg, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            block(base_channel*4, base_channel*8, 3, norm_cfg=norm_cfg, stride=1, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(base_channel*8, base_channel*8, norm_cfg=norm_cfg, indice_key='res3'),
            SparseBasicBlock(base_channel*8, base_channel*8, norm_cfg=norm_cfg, indice_key='res3'),
        )

        self.conv_out = spconv.SparseSequential(
            spconv.SubMConv3d(base_channel*8, out_channel, 3, padding=1),
            nn.GroupNorm(16, out_channel),
            nn.ReLU(inplace=True),
        )



    def forward(self, voxel_features, coors, batch_size):
        # spconv encoding
        coors = coors.int()  
        input_sp_tensor = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape_xyz[::-1], batch_size)

        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)

        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x = self.conv_out(x_conv3)

        # Convert the sparse tensor to a dense tensor
        x_dense = x.dense()

        # Permute the dimensions to bring the depth dimension last (if needed)
        x_dense = x_dense.permute(0, 1, 4, 3, 2)  # Adjust based on your specific dimension ordering

        # Apply a reduction operation to the depth dimension
        # Example: Using average pooling with a stride of 2 to reduce the depth dimension
        pool = nn.AvgPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2), padding=0)
        x = pool(x_dense)

        return {'x': x, 'pts_feats': [x]}

@MIDDLE_ENCODERS.register_module()
class SparsePolarRaDAREnc(nn.Module):
    def __init__(self, input_channel, norm_cfg, base_channel, out_channel, 
                sparse_shape_xyz, **kwargs):
        super().__init__()

        block = post_act_block
        self.sparse_shape_xyz = sparse_shape_xyz

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channel, base_channel, 1),
            nn.GroupNorm(16, base_channel),
            nn.ReLU(inplace=True))

        self.conv1 = spconv.SparseSequential(
            block(base_channel, base_channel*2, 1, norm_cfg=norm_cfg, stride=1, padding=0, indice_key='spconv1', conv_type='spconv'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, kernel_size=1, indice_key='res1'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, kernel_size=1, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            block(base_channel*2, base_channel*4, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(base_channel*4, base_channel*4, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res2'),
            SparseBasicBlock(base_channel*4, base_channel*4, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            block(base_channel*4, base_channel*8, 3, norm_cfg=norm_cfg, stride=1, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(base_channel*8, base_channel*8,  kernel_size=3, norm_cfg=norm_cfg, indice_key='res3'),
            SparseBasicBlock(base_channel*8, base_channel*8,  kernel_size=3, norm_cfg=norm_cfg, indice_key='res3'),
        )

        self.conv_out = spconv.SparseSequential(
            spconv.SubMConv3d(base_channel*8, out_channel, 3, padding=1),
            nn.GroupNorm(16, out_channel),
            nn.ReLU(inplace=True),
        )



    def forward(self, voxel_features, coors, batch_size):
        # spconv encoding
        coors = coors.int()  
        input_sp_tensor = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape_xyz[::-1], batch_size)

        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)

        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x = self.conv_out(x_conv3)

        # Convert the sparse tensor to a dense tensor
        x_dense = x.dense()

        # Permute the dimensions to bring the depth dimension last (if needed)
        x_dense = x_dense.permute(0, 1, 4, 3, 2)  # Adjust based on your specific dimension ordering

        # Apply a reduction operation to the depth dimension
        # Example: Using average pooling with a stride of 2 to reduce the depth dimension
        pool = nn.AvgPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2), padding=0)
        x = pool(x_dense)

        return {'x': x, 'pts_feats': [x]}
    


@MIDDLE_ENCODERS.register_module()
class RadarEncV2(nn.Module):
    def __init__(self, input_channel, norm_cfg, base_channel, out_channel, 
                sparse_shape_xyz, **kwargs):
        super().__init__()

        block = post_act_block
        self.sparse_shape_xyz = sparse_shape_xyz

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channel, base_channel, 1),
            nn.GroupNorm(8, base_channel),
            nn.ReLU(inplace=True))

        self.conv1 = spconv.SparseSequential(
            block(base_channel, base_channel*2, 3, norm_cfg=norm_cfg, stride=1, padding=1, indice_key='spconv1', conv_type='spconv'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, kernel_size=1, indice_key='res1'),
            block(base_channel*2, base_channel*2, 3, norm_cfg=norm_cfg, stride=1, padding=1, indice_key='spconv11', conv_type='spconv'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, kernel_size=3, indice_key='res12'),
            block(base_channel*2, base_channel*2, 3, norm_cfg=norm_cfg, stride=1, padding=1, indice_key='spconv12', conv_type='spconv'),

        )

        self.conv2 = spconv.SparseSequential(
            block(base_channel*2, base_channel*4, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(base_channel*4, base_channel*4, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res2'),
            block(base_channel*4, base_channel*8, 3, norm_cfg=norm_cfg, stride=1, padding=1, indice_key='spconv21', conv_type='spconv'),
            SparseBasicBlock(base_channel*8, base_channel*8, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res22'),

        )

        # self.conv3 = spconv.SparseSequential(
        #     block(base_channel*4, base_channel*8, 3, norm_cfg=norm_cfg, stride=1, padding=1, indice_key='spconv3', conv_type='spconv'),
        #     SparseBasicBlock(base_channel*8, base_channel*8,  kernel_size=3, norm_cfg=norm_cfg, indice_key='res3'),
        #     block(base_channel*8, base_channel*8, 3, norm_cfg=norm_cfg, stride=1, padding=1, indice_key='spconv31', conv_type='spconv'),
        #     SparseBasicBlock(base_channel*8, base_channel*8,  kernel_size=3, norm_cfg=norm_cfg, indice_key='res32'),
        #     block(base_channel*8, base_channel*8, 3, norm_cfg=norm_cfg, stride=1, padding=1, indice_key='spconv32', conv_type='spconv'),

        # )

        self.conv_out = spconv.SparseSequential(
            spconv.SubMConv3d(base_channel*8, out_channel, 3, padding=1),
            nn.GroupNorm(16, out_channel),
            nn.ReLU(inplace=True),
        )
        self.extra_conv = spconv.SparseSequential(
            spconv.SparseConv3d(
                out_channel, out_channel, (3, 1, 1), (2, 1, 1),padding=(1, 0, 0), bias=False
            ), 
            nn.GroupNorm(16, out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, voxel_features, coors, batch_size):
        # spconv encoding
        coors = coors.int()  
        input_sp_tensor = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape_xyz[::-1], batch_size)

        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        # x_conv3 = self.conv3(x_conv2)
        x = self.conv_out(x_conv2)

        x = self.extra_conv(x)
        x_dense = x.dense()

        # Permute the dimensions to bring the depth dimension last (if needed)
        x_dense = x_dense.permute(0, 1, 4, 3, 2)  # Adjust based on your specific dimension ordering
        return {'x': x_dense, 'pts_feats': [x]}    


@MIDDLE_ENCODERS.register_module()
class SparseRaDAREmb2x(nn.Module):
    def __init__(self, input_channel, norm_cfg, base_channel, out_channel, 
                sparse_shape_xyz, **kwargs):
        super().__init__()

        block = post_act_block
        self.sparse_shape_xyz = sparse_shape_xyz

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channel, base_channel, 3),
            nn.GroupNorm(16, base_channel),
            nn.ReLU(inplace=True))


        self.conv2 = spconv.SparseSequential(
            block(base_channel, base_channel*2, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, indice_key='res2'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, indice_key='res2'),
        )



        self.conv_out = spconv.SparseSequential(
            spconv.SubMConv3d(base_channel*2, out_channel, 3),
            nn.GroupNorm(16, out_channel),
            nn.ReLU(inplace=True))



    def forward(self, voxel_features, coors, batch_size):
        # spconv encoding
        coors = coors.int()  # zyx 与 points的输入(xyz)相反
        # FIXME bs=1 hardcode
        input_sp_tensor = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape_xyz[::-1], batch_size)

        x = self.conv_input(input_sp_tensor)

        x_conv2 = self.conv2(x)
        x = self.conv_out(x_conv2)


        return {'x': x.dense().permute(0,1,4,3,2), # B, C, W, H, D 
                'pts_feats': [x]}


class DopplerAttention(nn.Module):
    def __init__(self, doppler_size, hidden_size):
        super(DopplerAttention, self).__init__()
        self.query = nn.Linear(doppler_size, doppler_size, bias=False)
        self.key = nn.Linear(doppler_size, doppler_size, bias=False)
        self.value = nn.Linear(doppler_size, doppler_size, bias=False)
        self.doppler_size = doppler_size

    def forward(self, x):
        batch_size, doppler, range_dim, ele, azi = x.shape
        
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(batch_size, range_dim, -1, doppler)  # [batch_size, 256, 37*107, 64]

        # Process each range slice independently
        q = self.query(x)  # [batch_size, 256, 37*107, 64]
        k = self.key(x)    # [batch_size, 256, 37*107, 64]
        v = self.value(x)  # [batch_size, 256, 37*107, 64]

        # Reshape to [batch_size, 256, 64, 37*107] and compute attention
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2).transpose(-2, -1)  # Transpose for dot product
        attention_scores = torch.matmul(q, k)  # [batch_size, 256, 64, 64]
        attention_scores = attention_scores / (self.doppler_size ** 0.5)
        attention = F.softmax(attention_scores, dim=-1)

        # Apply attention to values
        v = v.permute(0, 1, 3, 2)
        attended_values = torch.matmul(attention, v)  # [batch_size, 256, 64, 37*107]
        
        # Reduce over Doppler and reshape back to original dimensions
        output = attended_values.sum(dim=2)  # [batch_size, 256, 37*107]
        output = output.view(batch_size, range_dim, ele, azi)  # [batch_size, 256, 37, 107]
        
        return output

@MIDDLE_ENCODERS.register_module()
class RadarEncV3(nn.Module):
    def __init__(self, input_channel, norm_cfg, base_channel, out_channel, 
                sparse_shape_xyz, **kwargs):
        super().__init__()

        block = post_act_block
        self.sparse_shape_xyz = sparse_shape_xyz

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channel, base_channel, 1),
            nn.GroupNorm(8, base_channel),
            nn.ReLU(inplace=True))

        self.conv1 = spconv.SparseSequential(
            block(base_channel, base_channel*2, 3, norm_cfg=norm_cfg, stride=1, padding=1, indice_key='spconv1', conv_type='spconv'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, kernel_size=1, indice_key='res1'),
            block(base_channel*2, base_channel*2, 3, norm_cfg=norm_cfg, stride=1, padding=1, indice_key='spconv11', conv_type='spconv'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, kernel_size=3, indice_key='res12'),

        )

        self.conv2 = spconv.SparseSequential(
            block(base_channel*2, base_channel*4, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(base_channel*4, base_channel*4, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res2'),
            block(base_channel*4, base_channel*8, 3, norm_cfg=norm_cfg, stride=1, padding=1, indice_key='spconv21', conv_type='spconv'),
            SparseBasicBlock(base_channel*8, base_channel*8, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res22'),

        )

        self.conv_out = spconv.SparseSequential(
            spconv.SubMConv3d(base_channel*8, out_channel, 3, padding=1),
            nn.GroupNorm(16, out_channel),
            nn.ReLU(inplace=True),
        )
        self.extra_conv = spconv.SparseSequential(
            spconv.SparseConv3d(
                out_channel, out_channel, (3, 1, 1), (2, 1, 1),padding=(1, 0, 0), bias=False
            ), 
            nn.GroupNorm(16, out_channel),
            nn.ReLU(inplace=True),
        )
        # self.dop_attn = DopplerAttention(64,32)

    def forward(self, radar_tensor):
      with torch.autograd.set_detect_anomaly(True):
        batch_size, d_dim, range_dim, elevation_dim, azithum_dim = radar_tensor.size()
        # radar_cube = self.dop_attn(radar_tensor)
        radar_cube = radar_tensor.mean(1)
        list_sparse_rdr_cubes = []
        list_sp_indices = []
        assert torch.isnan(radar_cube).sum().item() == 0
        for batch_idx in range(batch_size):
            cube = radar_cube[batch_idx]
            cube_flat = cube.reshape(cube.shape[0], -1)
            k = 100

            # Use 'torch.topk' to get the indices of the top k elements
            top_k_values, top_k_idx = torch.topk(cube_flat, k, dim=1)

            # Create a mask with the same shape as cube_flat
            mask_flat = torch.zeros_like(cube_flat, dtype=torch.bool)
            mask_flat[torch.arange(cube_flat.shape[0])[:, None], top_k_idx] = True
            mask = mask_flat.reshape(cube.shape)

            # Calculate the indices for the range, elevation, and azimuth
            range_inds = torch.arange(cube.shape[0])[:, None]
            elevation_inds = top_k_idx // cube.shape[2]
            azimuth_inds = top_k_idx % cube.shape[2]

            range_ind = torch.repeat_interleave(range_inds, k).flatten().cuda()
            elevation_ind = elevation_inds.flatten().cuda() 
            azimuth_ind = azimuth_inds.flatten().cuda() 
            # cube = torch.log10(cube)
            # normalized_cube = (cube - cube.min()) / (cube.max() - cube.min())
            # Extract the values at these indices from the original cube tensor
            power_val = cube[range_ind, elevation_ind, azimuth_ind].unsqueeze(-1)

            # Change the order to azimuth, range, elevation
            sparse_rdr_cube = torch.cat((azimuth_ind.float().unsqueeze(-1),
                                         range_ind.float().unsqueeze(-1),
                                         elevation_ind.float().unsqueeze(-1),
                                         power_val), dim=-1).to(torch.float32).cuda()
            list_sparse_rdr_cubes.append(sparse_rdr_cube)

            N, C = sparse_rdr_cube.shape
  
            batch_indices = torch.full((N, 1), batch_idx).cuda()
            azimuth_ind += 74
            elevation_ind += 1 
            sp_indices = torch.cat((batch_indices, elevation_ind.unsqueeze(-1), range_ind.unsqueeze(-1), azimuth_ind.unsqueeze(-1)), dim=-1)
            list_sp_indices.append(sp_indices)
        sparse_rdr_cube_all_batches = torch.cat(list_sparse_rdr_cubes, dim=0)
        sp_indices_all_batches = torch.cat(list_sp_indices, dim=0).cuda()
        
        assert torch.isnan(sparse_rdr_cube_all_batches).sum().item() == 0
        # spconv encoding
        coors = sp_indices_all_batches.int()  
        input_sp_tensor = spconv.SparseConvTensor(sparse_rdr_cube_all_batches, coors, self.sparse_shape_xyz[::-1], batch_size)

        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x = self.conv_out(x_conv2)

        x = self.extra_conv(x)
        x_dense = x.dense()

        # Permute the dimensions to bring the depth dimension last (if needed)
        x_dense = x_dense.permute(0, 1, 4, 3, 2)  # Adjust based on your specific dimension ordering
        return {'x': x_dense, 'pts_feats': [x]}    


class SparseDopplerAttention(nn.Module):
    def __init__(self, doppler_dim, num_ranges=256, max_ele=37, max_azi=107, embed_dim=16):
        super(SparseDopplerAttention, self).__init__()
        self.doppler_dim = doppler_dim
        self.embed_dim = embed_dim
        # Embeddings for elevation and azimuth
        self.elevation_embed = nn.Embedding(max_ele, embed_dim)
        self.azimuth_embed = nn.Embedding(max_azi, embed_dim)
        # Linear transformations for queries, keys, and values
        self.query = nn.Linear(doppler_dim + 2 * embed_dim, doppler_dim)
        self.key = nn.Linear(doppler_dim + 2 * embed_dim, doppler_dim)
        self.value = nn.Linear(doppler_dim + 2 * embed_dim, doppler_dim)

    def forward(self, power_vals, ele_indices,range_indices, azi_indices):
        # Calculate K assuming N = num_ranges * K
        num_ranges = 256
        K = power_vals.shape[0] // num_ranges

        # Prepare embedding indices based on original indices reshaped
        ele_embeddings = self.elevation_embed(ele_indices.reshape(num_ranges, K))  # [num_ranges, K, embed_dim]
        azi_embeddings = self.azimuth_embed(azi_indices.reshape(num_ranges, K))    # [num_ranges, K, embed_dim]

        # Reshape power_vals to [num_ranges, K, doppler_dim] and concatenate embeddings
        power_vals = power_vals.reshape(num_ranges, K,self.doppler_dim)
        power_vals = torch.cat([power_vals, ele_embeddings, azi_embeddings], dim=-1)  # [num_ranges, K, doppler_dim + 2*embed_dim]

        # Calculate Q, K, V
        Q = self.query(power_vals)
        K = self.key(power_vals).transpose(-2, -1)
        V = self.value(power_vals)

        # Compute attention scores and apply attention
        attention_scores = torch.matmul(Q, K) / (self.doppler_dim ** 0.5)
        attention = F.softmax(attention_scores, dim=-1)
        attended = torch.matmul(attention, V)  # [num_ranges, K, doppler_dim]

        # Sum over the Doppler dimension to reduce it
        attended_sum = torch.sum(attended, dim=-1)  # [num_ranges, K]
        power_values = torch.flatten(attended_sum,0)

        return power_values

@MIDDLE_ENCODERS.register_module()
class RadarEncV4(nn.Module):
    def __init__(self, input_channel, norm_cfg, base_channel, out_channel, 
                sparse_shape_xyz, **kwargs):
        super().__init__()

        block = post_act_block
        self.sparse_shape_xyz = sparse_shape_xyz

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channel, base_channel, 1),
            nn.GroupNorm(8, base_channel),
            nn.ReLU(inplace=True))

        self.conv1 = spconv.SparseSequential(
            block(base_channel, base_channel*2, 3, norm_cfg=norm_cfg, stride=1, padding=1, indice_key='spconv1', conv_type='spconv'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, kernel_size=1, indice_key='res1'),
            block(base_channel*2, base_channel*2, 3, norm_cfg=norm_cfg, stride=1, padding=1, indice_key='spconv11', conv_type='spconv'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, kernel_size=3, indice_key='res12'),

        )

        self.conv2 = spconv.SparseSequential(
            block(base_channel*2, base_channel*4, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(base_channel*4, base_channel*4, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res2'),
            block(base_channel*4, base_channel*8, 3, norm_cfg=norm_cfg, stride=1, padding=1, indice_key='spconv21', conv_type='spconv'),
            SparseBasicBlock(base_channel*8, base_channel*8, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res22'),

        )

        self.conv_out = spconv.SparseSequential(
            spconv.SubMConv3d(base_channel*8, out_channel, 3, padding=1),
            nn.GroupNorm(16, out_channel),
            nn.ReLU(inplace=True),
        )
        self.extra_conv = spconv.SparseSequential(
            spconv.SparseConv3d(
                out_channel, out_channel, (3, 1, 1), (2, 1, 1),padding=(1, 0, 0), bias=False
            ), 
            nn.GroupNorm(16, out_channel),
            nn.ReLU(inplace=True),
        )
        self.dop_attn = SparseDopplerAttention(doppler_dim=64)

    def forward(self, voxel_features, coors, batch_size):
        assert batch_size == 1 #

        radar_cube = self.dop_attn(voxel_features, coors[:,1], coors[:,2], coors[:,3])
        ele_ind = coors[:,1] + 1
        azi_ind = coors[:,3] + 74
        sparse_rdr_cube = torch.cat((ele_ind.float().unsqueeze(-1),
                                    coors[:,2].float().unsqueeze(-1),
                                    azi_ind.float().unsqueeze(-1),
                                    radar_cube.unsqueeze(-1)), dim=-1).to(torch.float32).cuda()

        assert torch.isnan(radar_cube).sum().item() == 0

        # spconv encoding
        coors = coors.int()  
        input_sp_tensor = spconv.SparseConvTensor(sparse_rdr_cube, coors, self.sparse_shape_xyz[::-1], batch_size)

        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x = self.conv_out(x_conv2)

        x = self.extra_conv(x)
        x_dense = x.dense()

        # Permute the dimensions to bring the depth dimension last (if needed)
        x_dense = x_dense.permute(0, 1, 4, 3, 2)  # Adjust based on your specific dimension ordering
        return {'x': x_dense, 'pts_feats': [x]}  
    
    
    

class RangeAttention(nn.Module):
    def __init__(self, num_ranges=175, K=5, embed_dim=16):
        super(RangeAttention, self).__init__()
        self.num_ranges = num_ranges
        self.K = K
        # Embeddings for elevation and azimuth
        self.elevation_embed = nn.Embedding(37, embed_dim)
        self.azimuth_embed = nn.Embedding(107, embed_dim)
        # Linear transformations for queries, keys, and values for the top 3 values
        self.query = nn.Linear(3 + 2 * embed_dim, 3)  # Just using the top 3 values
        self.key = nn.Linear(3 + 2 * embed_dim, 3)
        self.value = nn.Linear(3 + 2 * embed_dim, 3)

    def forward(self, power_vals, ele_indices, azi_indices):
        # Assume power_vals is [N, 8], where N = num_ranges * K
        # Reshape and extract parts
        power_vals = power_vals.view(self.num_ranges, self.K, 8)
        top_values = power_vals[:, :, :3]
        top_indices = power_vals[:, :, 3:6]  # Not used in attention
        mean_variance = power_vals[:, :, 6:]  # Not used in attention

        ele_embeddings = self.elevation_embed(ele_indices.view(self.num_ranges, self.K))
        azi_embeddings = self.azimuth_embed(azi_indices.view(self.num_ranges, self.K))

        # Concatenate embeddings with top 3 values
        x = torch.cat([top_values, ele_embeddings, azi_embeddings], dim=-1)  # [num_ranges, K, 3 + 2*embed_dim]

        # Calculate queries, keys, values
        queries = self.query(x)
        keys = self.key(x).transpose(-2, -1)  
        values = self.value(x)

        # Self-attention
        attention_scores = torch.matmul(queries, keys) / (3 ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_probs, values)

        # Combine attended values with other parts
        output = torch.cat([attended_values, top_indices, mean_variance], dim=2)  # Reconstruct the original structure

        return output.view(self.num_ranges, self.K, 8)




class MultiLayerRangeAttentionNoD(nn.Module):
    def __init__(self, num_layers=2, num_ranges=175, K=250, embed_dim=16):
        super(MultiLayerRangeAttentionNoD, self).__init__()
        self.num_layers = num_layers
        self.num_ranges = num_ranges
        self.K = K
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.ReLU()
        # Embeddings for elevation and azimuth
        self.elevation_embed = nn.Embedding(37, 3)
        self.azimuth_embed = nn.Embedding(107, 3)
        # Attention layers
        self.attention_layers = nn.ModuleList([
            SelfAttentionBlock(1+3+3,32,1).to(torch.float32)
            for _ in range(num_layers)
        ])

    def forward(self, power_vals, ele_indices, azi_indices):
        # Assume power_vals is [N, 1], where N = num_ranges * K
        self.k = power_vals.shape[0] // self.num_ranges
        mean = power_vals.view(self.num_ranges,  self.K,-1)
      

        ele_embeddings = self.elevation_embed(ele_indices.view(self.num_ranges, self.K)).to(torch.float32)
        azi_embeddings = self.azimuth_embed(azi_indices.view(self.num_ranges, self.K)).to(torch.float32)
         
        x = torch.cat([mean],-1)
        for layer in self.attention_layers:
            x = torch.cat([x,ele_embeddings,azi_embeddings],-1).to(torch.float32)
            x = layer(x)
            # x = self.dropout(self.norm(x))
            # x = out + x
            

        # Final output combines the refined attention values with the untouched parts
        output = torch.cat([x],-1)
        

        return output.view(self.num_ranges*self.K, 1)

class MultiLayerRangeAttention(nn.Module):
    def __init__(self, num_layers=2, num_ranges=175, K=250, embed_dim=16,doppler= 3):
        super(MultiLayerRangeAttention, self).__init__()
        self.num_layers = num_layers
        self.num_ranges = num_ranges
        self.K = K
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.ReLU()
        self.doppler = doppler
        # Embeddings for elevation and azimuth
        self.elevation_embed = nn.Embedding(37, 3)
        self.azimuth_embed = nn.Embedding(107, 3)
        self.doppler_embed = nn.Embedding(64,3)
        # Attention layers
        self.attention_layers = nn.ModuleList([
            SelfAttentionBlock(doppler+2+3+3+doppler*3,32,doppler+2).to(torch.float32)
            for _ in range(num_layers)
        ])

    def forward(self, power_vals, ele_indices, azi_indices):
        if self.doppler >= 4:
            max_doppler = 5
        else:
            max_doppler = 3
        # Assume power_vals is [N, 8], where N = num_ranges * K
        self.k = power_vals.shape[0] // self.num_ranges
        top_values = torch.log10(power_vals[:,:self.doppler]).view(self.num_ranges,  self.K,-1)
        top_indices = power_vals[:,max_doppler:max_doppler+self.doppler].view(self.num_ranges, self.K,-1).int()
        mean_variance = torch.log10(power_vals[:,-2:]).view(self.num_ranges,  self.K,-1)

        doppler_embeddings = self.doppler_embed(top_indices)
        doppler_embeddings = doppler_embeddings.flatten(2).to(torch.float32)

        ele_embeddings = self.elevation_embed(ele_indices.view(self.num_ranges, self.K)).to(torch.float32)
        azi_embeddings = self.azimuth_embed(azi_indices.view(self.num_ranges, self.K)).to(torch.float32)
        x = torch.cat([top_values,mean_variance],-1)
        for layer in self.attention_layers:
            x = torch.cat([x,ele_embeddings,azi_embeddings,doppler_embeddings],-1).to(torch.float32)

            x = layer(x)
            # x = self.dropout(self.norm(x))
            # x = out + x
            

        # Final output combines the refined attention values with the untouched parts
        output = torch.cat([x,top_indices],-1)
        

        return output.view(self.num_ranges*self.K, -1)

class MultiLayerRangeAttentionNEW(nn.Module):
    def __init__(self, num_layers=2, num_ranges=175, K=250, embed_dim=16):
        super(MultiLayerRangeAttention, self).__init__()
        self.num_layers = num_layers
        self.num_ranges = num_ranges
        self.K = K
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.ReLU()
        self.input_proj = nn.Linear(8,8)
        # Embeddings for elevation and azimuth
        self.elevation_embed = nn.Embedding(37, 3)
        self.azimuth_embed = nn.Embedding(107, 3)
        # Attention layers
        self.attention_layers = nn.ModuleList([
            SelfAttentionBlock(8+3+3,32,8).to(torch.float32)
            for _ in range(num_layers)
        ])

    def forward(self, power_vals, ele_indices, azi_indices):
        # Assume power_vals is [N, 8], where N = num_ranges * K
        self.k = power_vals.shape[0] // self.num_ranges
        
        top_values = power_vals[:,:3].view(self.num_ranges,  self.K,-1)
        top_indices = power_vals[:,3:6].view(self.num_ranges, self.K,-1).int()
        mean_variance = power_vals[:,6:].view(self.num_ranges,  self.K,-1)



        ele_embeddings = self.elevation_embed(ele_indices.view(self.num_ranges, self.K)).to(torch.float32)
        azi_embeddings = self.azimuth_embed(azi_indices.view(self.num_ranges, self.K)).to(torch.float32)
         
        x = torch.cat([top_values,top_indices,mean_variance],-1)
        x = self.input_proj(x)

        for layer in self.attention_layers:
            x = torch.cat([x,ele_embeddings,azi_embeddings],-1).to(torch.float32)
            x = layer(x)
            # x = self.dropout(self.norm(x))
            # x = out + x
            

        # Final output combines the refined attention values with the untouched parts
        output = x
        

        return output.view(self.num_ranges*self.K, 8)

class SelfAttentionBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SelfAttentionBlock, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x).transpose(-2, -1)
        values = self.value(x)

        attention_scores = torch.matmul(queries, keys) / (queries.size(-1) ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_probs, values)

        return attended_values

@MIDDLE_ENCODERS.register_module()
class RadarEncV5(nn.Module):
    def __init__(self, input_channel, norm_cfg, base_channel, out_channel, 
                sparse_shape_xyz, **kwargs):
        super().__init__()

        block = post_act_block
        self.sparse_shape_xyz = sparse_shape_xyz

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channel, base_channel, 1),
            nn.GroupNorm(8, base_channel),
            nn.ReLU(inplace=True))

        self.conv1 = spconv.SparseSequential(
            block(base_channel, base_channel*2, 3, norm_cfg=norm_cfg, stride=1, padding=1, indice_key='spconv1', conv_type='spconv'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, kernel_size=1, indice_key='res1'),
            block(base_channel*2, base_channel*2, 3, norm_cfg=norm_cfg, stride=1, padding=1, indice_key='spconv11', conv_type='spconv'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, kernel_size=3, indice_key='res12'),

        )

        self.conv2 = spconv.SparseSequential(
            block(base_channel*2, base_channel*4, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(base_channel*4, base_channel*4, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res2'),
            block(base_channel*4, base_channel*8, 3, norm_cfg=norm_cfg, stride=1, padding=1, indice_key='spconv21', conv_type='spconv'),
            SparseBasicBlock(base_channel*8, base_channel*8, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res22'),

        )

        self.conv_out = spconv.SparseSequential(
            spconv.SubMConv3d(base_channel*8, out_channel, 3, padding=1),
            nn.GroupNorm(16, out_channel),
            nn.ReLU(inplace=True),
        )
        # self.extra_conv = spconv.SparseSequential(
        #     spconv.SparseConv3d(
        #         out_channel, out_channel, (3, 1, 1), (2, 1, 1),padding=(1, 0, 0), bias=False
        #     ), 
        #     nn.GroupNorm(16, out_channel),
        #     nn.ReLU(inplace=True),
        # )
        # self.range_attn = MultiLayerRangeAttention(num_layers=4,K=200)

    def forward(self, voxel_features, coors, batch_size):
        assert batch_size == 1 #

        # radar_cube = self.range_attn(voxel_features, coors[:,1],  coors[:,3])
        ele_ind = coors[:,1] + 1
        range_ind = coors[:,2] + (256-175)//2
        azi_ind = coors[:,3] + 74
        sparse_rdr_cube = torch.cat((ele_ind.float().unsqueeze(-1),
                                    range_ind.float().unsqueeze(-1),
                                    azi_ind.float().unsqueeze(-1),
                                    voxel_features), dim=-1).to(torch.float32).cuda()


        # spconv encoding
        coors = coors.int()  
        input_sp_tensor = spconv.SparseConvTensor(sparse_rdr_cube, coors, self.sparse_shape_xyz[::-1], batch_size)

        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x = self.conv_out(x_conv2)

        # x = self.extra_conv(x)
        x_dense = x.dense()

        # Permute the dimensions to bring the depth dimension last (if needed)
        x_dense = x_dense.permute(0, 1, 4, 3, 2)  # Adjust based on your specific dimension ordering
        return {'x': x_dense, 'pts_feats': [x]}  
    
@MIDDLE_ENCODERS.register_module()
class RadarEncV6(nn.Module):
    def __init__(self, input_channel, norm_cfg, base_channel, out_channel, 
                sparse_shape_xyz, top_k=250,**kwargs):
        super().__init__()

        block = post_act_block
        self.sparse_shape_xyz = sparse_shape_xyz

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channel, base_channel, 1),
            nn.GroupNorm(8, base_channel),
            nn.ReLU(inplace=True))

        self.conv1 = spconv.SparseSequential(
            block(base_channel, base_channel*2, 3, norm_cfg=norm_cfg, stride=1, padding=1, indice_key='spconv1', conv_type='spconv'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, kernel_size=3, indice_key='res1'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, kernel_size=3, indice_key='res1'),

        )
        self.conv2 = spconv.SparseSequential(
            block(base_channel*2, base_channel*4, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(base_channel*4, base_channel*4, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res2'),
            SparseBasicBlock(base_channel*4, base_channel*4, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res2'),

        )
        self.conv3 = spconv.SparseSequential(
            block(base_channel*4, base_channel*8, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(base_channel*8, base_channel*8, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res3'),
            SparseBasicBlock(base_channel*8, base_channel*8, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res3'),

        )

        self.conv_out = spconv.SparseSequential(
            spconv.SubMConv3d(base_channel*8, out_channel, 3, padding=1),
            nn.GroupNorm(16, out_channel),
            nn.ReLU(inplace=True),
        )

        self.range_attn = MultiLayerRangeAttentionV2(num_layers=4,K=top_k,num_ranges=200)

    def forward(self, voxel_features, coors, batch_size):
        assert batch_size == 1 #

        ele_ind = coors[:,1] 
        range_ind = coors[:,2] 
        azi_ind = coors[:,3] 
        voxel_features = self.range_attn(voxel_features, coors[:,1],  coors[:,3])
        # sparse_rdr_cube = torch.cat((ele_ind.float().unsqueeze(-1),
        #                             range_ind.float().unsqueeze(-1),
        #                             azi_ind.float().unsqueeze(-1),
        #                             voxel_features), dim=-1).to(torch.float32).cuda()

        # coors = torch.cat([coors[:,0].unsqueeze(-1),ele_ind.unsqueeze(-1),azi_ind.unsqueeze(-1),range_ind.unsqueeze(-1)],-1)
        # spconv encoding
        coors_padded = torch.cat([coors[:,0].unsqueeze(-1),ele_ind.unsqueeze(-1),range_ind.unsqueeze(-1),azi_ind.unsqueeze(-1)],-1)

        coors_padded = coors_padded.int()  
        sparse_rdr_cube = torch.cat((ele_ind.float().unsqueeze(-1),
                                    range_ind.float().unsqueeze(-1),
                                    azi_ind.float().unsqueeze(-1),
                                    voxel_features), dim=-1).to(torch.float32).cuda()
        input_sp_tensor = spconv.SparseConvTensor(sparse_rdr_cube.to(torch.float32), coors_padded, self.sparse_shape_xyz[::-1], batch_size)

        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x = self.conv2(x_conv1)
        x = self.conv3(x)
        x = self.conv_out(x)
        x_dense = x.dense()

        x_dense = x_dense.permute(0, 1, 4, 3, 2)  
        return {'x': x_dense, 'pts_feats': [x]}  


class MultiLayerRangeAttentionV2(nn.Module):
    def __init__(self, num_layers=2, num_ranges=256, K=250, embed_dim=8, num_heads=4):
        super(MultiLayerRangeAttentionV2, self).__init__()
        self.num_layers = num_layers
        self.num_ranges = num_ranges
        self.K = K
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Embeddings for elevation and azimuth
        self.values_project = nn.Linear(8,embed_dim).to(torch.float32)
        self.azimuth_embed = nn.Embedding(107, embed_dim//2)
        self.elevation_embed = nn.Embedding(37, embed_dim//2)

        # Attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim * 2, num_heads, dropout=0.1).to(torch.float32)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(embed_dim * 2)
        
    def forward(self, power_vals, ele_indices, azi_indices):
        # Assume power_vals is [N, 8], where N = num_ranges * K
        self.k = power_vals.shape[0] // self.num_ranges
        power_vals = power_vals.view(self.num_ranges, self.K,-1)
        top_values = self.values_project(power_vals.to(torch.float32))
        ele_embeddings = self.elevation_embed(ele_indices.view(self.num_ranges, self.K)).to(torch.float32)
        azi_embeddings = self.azimuth_embed(azi_indices.view(self.num_ranges, self.K)).to(torch.float32)
        
        x = torch.cat([top_values, ele_embeddings, azi_embeddings], -1).to(torch.float32)
        for layer in self.attention_layers:
            x = x.permute(1, 0, 2)  # [K, num_ranges, embed_dim * 4]
            attn_output, _ = layer(x, x, x)
            x = self.norm(attn_output + x)
            x = self.dropout(x)
            x = x.permute(1, 0, 2)  # [num_ranges, K, embed_dim * 4]

        # Final output combines the refined attention values with the untouched parts
        output = x
        return output.reshape(self.num_ranges * self.K, -1)
@MIDDLE_ENCODERS.register_module()
class RadarEncV7(nn.Module):
    def __init__(self, input_channel, norm_cfg, base_channel, out_channel, 
                sparse_shape_xyz, **kwargs):
        super().__init__()

        block = post_act_block
        self.sparse_shape_xyz = sparse_shape_xyz

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channel, base_channel, 1),
            nn.GroupNorm(8, base_channel),
            nn.ReLU(inplace=True))

        self.conv1 = spconv.SparseSequential(
            block(base_channel, base_channel*2, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv1', conv_type='spconv'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, kernel_size=3, indice_key='res1'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, kernel_size=3, indice_key='res1'),

        )
        self.conv2 = spconv.SparseSequential(
            block(base_channel*2, base_channel*4, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(base_channel*4, base_channel*4, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res2'),
            SparseBasicBlock(base_channel*4, base_channel*4, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res2'),

        )
        self.conv3 = spconv.SparseSequential(
            block(base_channel*4, base_channel*8, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(base_channel*8, base_channel*8, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res3'),
            SparseBasicBlock(base_channel*8, base_channel*8, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res3'),

        )

        self.conv_out = spconv.SparseSequential(
            spconv.SubMConv3d(base_channel*8, out_channel, 3, padding=1),
            nn.GroupNorm(16, out_channel),
            nn.ReLU(inplace=True),
        )

        self.range_attn = MultiLayerRangeAttentionNEW(num_layers=4,num_ranges=175)

    def forward(self, voxel_features, coors, batch_size):
        assert batch_size == 1 #

        range_indices = coors[:,2]
        elevation_indices = coors[:,1]
        azimuth_indices =coors[:,3]

        range_resolution = 0.46
        elevation_resolution = torch.deg2rad(torch.tensor(1.0)).cuda() 
        azimuth_resolution = torch.deg2rad(torch.tensor(1.0)).cuda()   
        grid_size = 0.05

        # ROI bounds
        x_min, x_max = 0, 51.2
        y_min, y_max = -25.6, 25.6
        z_min, z_max = -2.6, 3


        # Radar tensor shape and center indices
        _, range_bins, ele_bins, azi_bins = (1, 175, 37, 107)
        ele_center = ele_bins // 2
        azi_center = azi_bins // 2


        # Convert spherical to Cartesian coordinates
        ranges = range_indices * range_resolution
        elevations = (elevation_indices - ele_center) * elevation_resolution
        azimuths = (azimuth_indices - azi_center) * azimuth_resolution

        x = ranges * torch.cos(elevations) * torch.cos(azimuths)
        y = ranges * torch.cos(elevations) * torch.sin(azimuths)
        z = ranges * torch.sin(elevations)

        valid_indices = (x_min <= x) & (x <= x_max) & (y_min <= y) & (y <= y_max) & (z_min <= z) & (z <= z_max)
        x = x[valid_indices]
        y = y[valid_indices]
        z = z[valid_indices]

        # Convert Cartesian coordinates to grid indices
        i_x = torch.floor((x - x_min) / grid_size).to(torch.int)
        i_y = torch.floor((y - y_min) / grid_size).to(torch.int)
        i_z = torch.floor((z - z_min) / grid_size).to(torch.int)

        voxel_features = self.range_attn(voxel_features, coors[:,1],  coors[:,3])
        sparse_rdr_cube = torch.cat((i_z.float().unsqueeze(-1),
                                    i_y.float().unsqueeze(-1),
                                    i_x.float().unsqueeze(-1),
                                    voxel_features[valid_indices]), dim=-1).to(torch.float32).cuda()
        # coors = torch.cat([coors[:,0].unsqueeze(-1),ele_ind.unsqueeze(-1),azi_ind.unsqueeze(-1),range_ind.unsqueeze(-1)],-1)
        # spconv encoding
        coors_padded = torch.cat([coors[valid_indices,0].unsqueeze(-1),i_z.unsqueeze(-1),i_y.unsqueeze(-1),i_x.unsqueeze(-1)],-1)
        coors_padded = coors_padded.int()  
        input_sp_tensor = spconv.SparseConvTensor(sparse_rdr_cube, coors_padded, self.sparse_shape_xyz[::-1], batch_size)

        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x = self.conv2(x_conv1)
        x = self.conv3(x)
        x = self.conv_out(x)
        x_dense = x.dense()

        x_dense = x_dense.permute(0, 1, 4, 3, 2)  
        return {'x': x_dense, 'pts_feats': [x]}  
    


@MIDDLE_ENCODERS.register_module()
class RadarEncV8(nn.Module):
    def __init__(self, input_channel, norm_cfg, base_channel, out_channel, 
                sparse_shape_xyz, top_k=250,doppler = 3,**kwargs):
        super().__init__()
        self.record_time = False
        block = post_act_block
        self.sparse_shape_xyz = sparse_shape_xyz

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(2*doppler+2+3, base_channel, 1),
            nn.GroupNorm(8, base_channel),
            nn.ReLU(inplace=True))

        self.conv1 = spconv.SparseSequential(
            block(base_channel, base_channel*2, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv1', conv_type='spconv'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, kernel_size=3, indice_key='res1'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, kernel_size=3, indice_key='res1'),

        )
        self.conv2 = spconv.SparseSequential(
            block(base_channel*2, base_channel*8, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(base_channel*8, base_channel*8, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res2'),
            SparseBasicBlock(base_channel*8, base_channel*8, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res2'),

        )

        self.conv_out = spconv.SparseSequential(
            spconv.SubMConv3d(base_channel*8, out_channel, 3, padding=1),
            nn.GroupNorm(16, out_channel),
            nn.ReLU(inplace=True),
        )

        self.range_attn = MultiLayerRangeAttention(num_layers=4,K=top_k,num_ranges=175,doppler=doppler)

    def forward(self, voxel_features, coors, batch_size):
        assert batch_size == 1 #
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        ele_ind = coors[:,1]*2 + (56-37*2)//2
        range_ind = coors[:,2]  * 2
        azi_ind = coors[:,3] * 2 + (512-107*2)//2
        voxel_features = self.range_attn(voxel_features, coors[:,1],  coors[:,3])
        sparse_rdr_cube = torch.cat((ele_ind.float().unsqueeze(-1),
                                    range_ind.float().unsqueeze(-1),
                                    azi_ind.float().unsqueeze(-1),
                                    voxel_features), dim=-1).to(torch.float32).cuda()
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            print(f"range_attn: {t1-t0}")
            t0 = time.time()

        # coors = torch.cat([coors[:,0].unsqueeze(-1),ele_ind.unsqueeze(-1),azi_ind.unsqueeze(-1),range_ind.unsqueeze(-1)],-1)
        # spconv encoding
        
        coors_padded = torch.cat([coors[:,0].unsqueeze(-1),ele_ind.unsqueeze(-1),range_ind.unsqueeze(-1),azi_ind.unsqueeze(-1)],-1)

        coors_padded = coors_padded.int()  
        sparse_rdr_cube = torch.cat((ele_ind.float().unsqueeze(-1),
                                    range_ind.float().unsqueeze(-1),
                                    azi_ind.float().unsqueeze(-1),
                                    voxel_features), dim=-1).to(torch.float32).cuda()
        input_sp_tensor = spconv.SparseConvTensor(sparse_rdr_cube.to(torch.float32), coors_padded, self.sparse_shape_xyz[::-1], batch_size)

        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x = self.conv2(x_conv1)
        x = self.conv_out(x)
        x_dense = x.dense()

        x_dense = x_dense.permute(0, 1, 4, 3, 2) 
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            print(f"sparse_conv: {t1-t0}")
        return {'x': x_dense, 'pts_feats': [x]}  


@MIDDLE_ENCODERS.register_module()
class RadarEncV8small(nn.Module):
    def __init__(self, input_channel, norm_cfg, base_channel, out_channel, 
                sparse_shape_xyz, top_k=250,**kwargs):
        super().__init__()
        self.record_time = True
        block = post_act_block
        self.sparse_shape_xyz = sparse_shape_xyz

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channel, base_channel, 1),
            nn.GroupNorm(8, base_channel),
            nn.ReLU(inplace=True))

        self.conv1 = spconv.SparseSequential(
            block(base_channel, base_channel*2, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv1', conv_type='spconv'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, kernel_size=3, indice_key='res1'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, kernel_size=3, indice_key='res1'),

        )
        self.conv2 = spconv.SparseSequential(
            block(base_channel*2, base_channel*4, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(base_channel*4, base_channel*4, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res2'),
            SparseBasicBlock(base_channel*4, base_channel*4, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res2'),

        )

        self.conv_out = spconv.SparseSequential(
            spconv.SubMConv3d(base_channel*4, out_channel, 3, padding=1),
            nn.GroupNorm(16, out_channel),
            nn.ReLU(inplace=True),
        )

        self.range_attn = MultiLayerRangeAttention(num_layers=4,K=top_k,num_ranges=175)

    def forward(self, voxel_features, coors, batch_size):
        assert batch_size == 1 #
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()

        ele_ind = coors[:,1]*2 + (56-37*2)//2
        range_ind = coors[:,2]  * 2
        azi_ind = coors[:,3] * 2 + (512-107*2)//2
        voxel_features = self.range_attn(voxel_features, coors[:,1],  coors[:,3])
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            print(f"range_attn: {t1-t0}")
            t0 = time.time()
        coors_padded = torch.cat([coors[:,0].unsqueeze(-1),ele_ind.unsqueeze(-1),range_ind.unsqueeze(-1),azi_ind.unsqueeze(-1)],-1)

        coors_padded = coors_padded.int()  
        sparse_rdr_cube = torch.cat((ele_ind.float().unsqueeze(-1),
                                range_ind.float().unsqueeze(-1),
                                azi_ind.float().unsqueeze(-1),
                                voxel_features), dim=-1).to(torch.float16).cuda()
        input_sp_tensor = spconv.SparseConvTensor(sparse_rdr_cube.to(torch.half), coors_padded, self.sparse_shape_xyz[::-1], batch_size)

        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x = self.conv2(x_conv1)
        x = self.conv_out(x)
        x_dense = x.dense()
        x_dense = x_dense.permute(0, 1, 4, 3, 2) 
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            print(f"sparse_conv: {t1-t0}")
        return {'x': x_dense, 'pts_feats': [x]}  




@MIDDLE_ENCODERS.register_module()
class RadarEncNO(nn.Module):
    def __init__(self, input_channel, norm_cfg, base_channel, out_channel, 
                sparse_shape_xyz, top_k=250,**kwargs):
        super().__init__()

        block = post_act_block
        self.sparse_shape_xyz = sparse_shape_xyz

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channel, base_channel, 1),
            nn.GroupNorm(8, base_channel),
            nn.ReLU(inplace=True))

        self.conv1 = spconv.SparseSequential(
            block(base_channel, base_channel*2, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv1', conv_type='spconv'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, kernel_size=3, indice_key='res1'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, kernel_size=3, indice_key='res1'),

        )
        self.conv2 = spconv.SparseSequential(
            block(base_channel*2, base_channel*8, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(base_channel*8, base_channel*8, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res2'),
            SparseBasicBlock(base_channel*8, base_channel*8, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res2'),

        )

        self.conv_out = spconv.SparseSequential(
            spconv.SubMConv3d(base_channel*8, out_channel, 3, padding=1),
            nn.GroupNorm(16, out_channel),
            nn.ReLU(inplace=True),
        )

        self.range_attn = MultiLayerRangeAttentionNoD(num_layers=4,K=top_k,num_ranges=175)

    def forward(self, voxel_features, coors, batch_size):
        assert batch_size == 1 #

        ele_ind = coors[:,1]*2 + (56-37*2)//2
        range_ind = coors[:,2]  * 2
        azi_ind = coors[:,3] * 2 + (512-107*2)//2
        voxel_features = torch.log10(voxel_features[:,-2]) #mean only
        voxel_features =voxel_features.unsqueeze(-1)
        voxel_features = self.range_attn(voxel_features, coors[:,1],  coors[:,3])
        sparse_rdr_cube = torch.cat((ele_ind.float().unsqueeze(-1),
                                    range_ind.float().unsqueeze(-1),
                                    azi_ind.float().unsqueeze(-1),
                                    voxel_features), dim=-1).to(torch.float32).cuda()

        # coors = torch.cat([coors[:,0].unsqueeze(-1),ele_ind.unsqueeze(-1),azi_ind.unsqueeze(-1),range_ind.unsqueeze(-1)],-1)
        # spconv encoding
        coors_padded = torch.cat([coors[:,0].unsqueeze(-1),ele_ind.unsqueeze(-1),range_ind.unsqueeze(-1),azi_ind.unsqueeze(-1)],-1)

        coors_padded = coors_padded.int()  
        sparse_rdr_cube = torch.cat((ele_ind.float().unsqueeze(-1),
                                    range_ind.float().unsqueeze(-1),
                                    azi_ind.float().unsqueeze(-1),
                                    voxel_features), dim=-1).to(torch.float32).cuda()
        input_sp_tensor = spconv.SparseConvTensor(sparse_rdr_cube.to(torch.float32), coors_padded, self.sparse_shape_xyz[::-1], batch_size)

        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x = self.conv2(x_conv1)
        x = self.conv_out(x)
        x_dense = x.dense()

        x_dense = x_dense.permute(0, 1, 4, 3, 2)  
        return {'x': x_dense, 'pts_feats': [x]}  

@MIDDLE_ENCODERS.register_module()
class RadarEncV9(nn.Module):
    def __init__(self, input_channel, norm_cfg, base_channel, out_channel, 
                sparse_shape_xyz, top_k=250,**kwargs):
        super().__init__()

        block = post_act_block
        self.sparse_shape_xyz = sparse_shape_xyz

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channel, base_channel, 1),
            nn.GroupNorm(8, base_channel),
            nn.ReLU(inplace=True))

        self.conv1 = spconv.SparseSequential(
            block(base_channel, base_channel*2, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv1', conv_type='spconv'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, kernel_size=3, indice_key='res1'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, kernel_size=3, indice_key='res1'),

        )
        self.conv2 = spconv.SparseSequential(
            block(base_channel*2, base_channel*8, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(base_channel*8, base_channel*8, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res2'),
            SparseBasicBlock(base_channel*8, base_channel*8, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res2'),

        )

        self.conv_out = spconv.SparseSequential(
            spconv.SubMConv3d(base_channel*8, out_channel, 3, padding=1),
            nn.GroupNorm(16, out_channel),
            nn.ReLU(inplace=True),
        )

        self.range_attn = MultiLayerRangeAttentionV2(num_layers=2,K=top_k,num_ranges=175)

    def forward(self, voxel_features, coors, batch_size):
        assert batch_size == 1 #

        ele_ind = coors[:,1]*2 + (56-37*2)//2
        range_ind = coors[:,2]  * 2
        azi_ind = coors[:,3] * 2 + (512-107*2)//2
        voxel_features = self.range_attn(voxel_features, coors[:,1],  coors[:,3])
        sparse_rdr_cube = torch.cat((ele_ind.float().unsqueeze(-1),
                                    range_ind.float().unsqueeze(-1),
                                    azi_ind.float().unsqueeze(-1),
                                    voxel_features), dim=-1).to(torch.float32).cuda()

        # coors = torch.cat([coors[:,0].unsqueeze(-1),ele_ind.unsqueeze(-1),azi_ind.unsqueeze(-1),range_ind.unsqueeze(-1)],-1)
        # spconv encoding
        coors_padded = torch.cat([coors[:,0].unsqueeze(-1),ele_ind.unsqueeze(-1),range_ind.unsqueeze(-1),azi_ind.unsqueeze(-1)],-1)

        coors_padded = coors_padded.int()  
        sparse_rdr_cube = torch.cat((ele_ind.float().unsqueeze(-1),
                                    range_ind.float().unsqueeze(-1),
                                    azi_ind.float().unsqueeze(-1),
                                    voxel_features), dim=-1).to(torch.float32).cuda()
        input_sp_tensor = spconv.SparseConvTensor(sparse_rdr_cube.to(torch.float32), coors_padded, self.sparse_shape_xyz[::-1], batch_size)

        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x = self.conv2(x_conv1)
        x = self.conv_out(x)
        x_dense = x.dense()

        x_dense = x_dense.permute(0, 1, 4, 3, 2)  
        return {'x': x_dense, 'pts_feats': [x]}  





@MIDDLE_ENCODERS.register_module()
class RadarEncC(nn.Module):
    def __init__(self, input_channel, norm_cfg, base_channel, out_channel, 
                sparse_shape_xyz, **kwargs):
        super().__init__()

        block = post_act_block
        self.sparse_shape_xyz = sparse_shape_xyz

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channel, base_channel, 1),
            nn.GroupNorm(8, base_channel),
            nn.ReLU(inplace=True))

        self.conv1 = spconv.SparseSequential(
            block(base_channel, base_channel*2, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv1', conv_type='spconv'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, kernel_size=3, indice_key='res1'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, kernel_size=3, indice_key='res1'),

        )
        self.conv2 = spconv.SparseSequential(
            block(base_channel*2, base_channel*4, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(base_channel*4, base_channel*4, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res2'),
            SparseBasicBlock(base_channel*4, base_channel*4, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res2'),

        )
        self.conv3 = spconv.SparseSequential(
            block(base_channel*4, base_channel*8, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(base_channel*8, base_channel*8, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res3'),
            SparseBasicBlock(base_channel*8, base_channel*8, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res3'),

        )

        self.conv_out = spconv.SparseSequential(
            spconv.SubMConv3d(base_channel*8, out_channel, 3, padding=1),
            nn.GroupNorm(16, out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, voxel_features, coors, batch_size):

        range_indices = coors[:,2]
        elevation_indices = coors[:,1]
        azimuth_indices =coors[:,3]

        range_resolution = 0.46
        elevation_resolution = torch.deg2rad(torch.tensor(1.0)).cuda() 
        azimuth_resolution = torch.deg2rad(torch.tensor(1.0)).cuda()   
        grid_size = 0.4

        # ROI bounds
        x_min, x_max = 0, 51.2
        y_min, y_max = -25.6, 25.6
        z_min, z_max = -2.6, 3


        # Radar tensor shape and center indices
        _, range_bins, ele_bins, azi_bins = (1, 175, 37, 107)
        ele_center = ele_bins // 2
        azi_center = azi_bins // 2


        # Convert spherical to Cartesian coordinates
        ranges = range_indices * range_resolution
        elevations = (elevation_indices - ele_center) * elevation_resolution
        azimuths = (azimuth_indices - azi_center) * azimuth_resolution

        x = ranges * torch.cos(elevations) * torch.cos(azimuths)
        y = ranges * torch.cos(elevations) * torch.sin(azimuths)
        z = ranges * torch.sin(elevations)

        valid_indices = (x_min <= x) & (x <= x_max) & (y_min <= y) & (y <= y_max) & (z_min <= z) & (z <= z_max)
        x = x[valid_indices]
        y = y[valid_indices]
        z = z[valid_indices]

        # Convert Cartesian coordinates to grid indices
        i_x = torch.floor((x - x_min) / grid_size).to(torch.int) * 8
        i_y = torch.floor((y - y_min) / grid_size).to(torch.int) * 8
        i_z = torch.floor((z - z_min) / grid_size).to(torch.int) * 8

        sparse_rdr_cube = torch.cat((i_z.float().unsqueeze(-1),
                                    i_y.float().unsqueeze(-1),
                                    i_x.float().unsqueeze(-1),
                                    voxel_features[valid_indices]), dim=-1).to(torch.float32).cuda()
        # coors = torch.cat([coors[:,0].unsqueeze(-1),ele_ind.unsqueeze(-1),azi_ind.unsqueeze(-1),range_ind.unsqueeze(-1)],-1)
        # spconv encoding
        coors_padded = torch.cat([coors[valid_indices,0].unsqueeze(-1),i_z.unsqueeze(-1),i_y.unsqueeze(-1),i_x.unsqueeze(-1)],-1)
        coors_padded = coors_padded.int()  
        input_sp_tensor = spconv.SparseConvTensor(sparse_rdr_cube, coors_padded, self.sparse_shape_xyz[::-1], batch_size)

        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x = self.conv2(x_conv1)
        x = self.conv3(x)
        x = self.conv_out(x)
        x_dense = x.dense()

        x_dense = x_dense.permute(0, 1, 4, 3, 2)  
        return {'x': x_dense, 'pts_feats': [x]}  
    


@MIDDLE_ENCODERS.register_module()
class RadarEncVS(nn.Module):
    def __init__(self, input_channel, norm_cfg, base_channel, out_channel, 
                sparse_shape_xyz, top_k=250,**kwargs):
        super().__init__()

        block = post_act_block
        self.sparse_shape_xyz = sparse_shape_xyz

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channel, base_channel, 1),
            nn.GroupNorm(8, base_channel),
            nn.ReLU(inplace=True))

        self.conv1 = spconv.SparseSequential(
            block(base_channel, base_channel*2, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv1', conv_type='spconv'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, kernel_size=3, indice_key='res1'),
            SparseBasicBlock(base_channel*2, base_channel*2, norm_cfg=norm_cfg, kernel_size=3, indice_key='res1'),

        )
        self.conv2 = spconv.SparseSequential(
            block(base_channel*2, base_channel*8, 3, norm_cfg=norm_cfg, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(base_channel*8, base_channel*8, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res2'),
            SparseBasicBlock(base_channel*8, base_channel*8, norm_cfg=norm_cfg,  kernel_size=3, indice_key='res2'),

        )

        self.conv_out = spconv.SparseSequential(
            spconv.SubMConv3d(base_channel*8, out_channel, 3, padding=1),
            nn.GroupNorm(16, out_channel),
            nn.ReLU(inplace=True),
        )


    def forward(self, voxel_features, coors, batch_size):
        assert batch_size == 1 #

        ele_ind = coors[:,1]*2 + (56-37*2)//2
        range_ind = coors[:,2]  * 2
        azi_ind = coors[:,3] * 2 + (512-107*2)//2
        sparse_rdr_cube = torch.cat((ele_ind.float().unsqueeze(-1),
                                    range_ind.float().unsqueeze(-1),
                                    azi_ind.float().unsqueeze(-1),
                                    voxel_features), dim=-1).to(torch.float32).cuda()

        # coors = torch.cat([coors[:,0].unsqueeze(-1),ele_ind.unsqueeze(-1),azi_ind.unsqueeze(-1),range_ind.unsqueeze(-1)],-1)
        # spconv encoding
        coors_padded = torch.cat([coors[:,0].unsqueeze(-1),ele_ind.unsqueeze(-1),range_ind.unsqueeze(-1),azi_ind.unsqueeze(-1)],-1)

        coors_padded = coors_padded.int()  
        sparse_rdr_cube = torch.cat((ele_ind.float().unsqueeze(-1),
                                    range_ind.float().unsqueeze(-1),
                                    azi_ind.float().unsqueeze(-1),
                                    voxel_features), dim=-1).to(torch.float32).cuda()
        input_sp_tensor = spconv.SparseConvTensor(sparse_rdr_cube.to(torch.float32), coors_padded, self.sparse_shape_xyz[::-1], batch_size)

        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x = self.conv2(x_conv1)
        x = self.conv_out(x)
        x_dense = x.dense()

        x_dense = x_dense.permute(0, 1, 4, 3, 2)  
        return {'x': x_dense, 'pts_feats': [x]}  

