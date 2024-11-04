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



