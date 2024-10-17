_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
input_modality = dict(
    use_lidar=False,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
plugin = True
plugin_dir = "projects/occ_plugin/"
img_norm_cfg = None
occ_path = "/mnt/data/DataSet/K-RadarOOC/train"
train_ann_file = "/home/xiangyu/SurroundOcc/kradar_dict_train_doppler8.pkl"
val_ann_file = "/home/xiangyu/SurroundOcc/kradar_dict_val_doppler8.pkl"
# For nuScenes we usually do 10-class detection
class_names = LIST_CLS_NAME = [
    'Static',
    'Sedan',
    'Bus or Truck',
    'Motorcycle',
    'Bicycle',
    'Pedestrian',
    'Pedestrian Group',
    'Bicycle Group',
    'Unknow'
] 
point_cloud_range = [0, -25.6, -2.6, 51.2, 25.6, 3.0]
occ_size = [128, 128, 14]
voxel_channels = [80, 160, 320, 640]
empty_idx = 0  # noise 0-->255
num_cls = 3 # 
visible_mask = False

cascade_ratio = 1
sample_from_voxel = False
sample_from_img = False

dataset_type = 'NuscOCCDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')


numC_Trans = _dim_= 192
voxel_out_channel = 256
voxel_out_indices = (0, 1, 2, 3)
_pos_dim_ = _dim_//3
_ffn_dim_ = _dim_*2
_num_cams_ = 5
_num_layers_self_ = 2
_num_layers_cross_ = 2
_num_points_self_ = 8
top_k = 250
doppler = 1
model = dict(
    type='RadarDopplerAttnEASelf',
    loss_norm=True,
    embed_dims = numC_Trans,
    top_k = top_k,
    pts_middle_encoder=dict(
        type='RadarEncV8',
        input_channel=11,
        base_channel=32,
        out_channel=numC_Trans,
        top_k = top_k,
        doppler = doppler,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        sparse_shape_xyz=[512, 512, 56],  
        ),
    occ_encoder_backbone=dict(
        type='CustomResNet3D',
        depth=18,
        n_input_channels=numC_Trans,
        block_inplanes=voxel_channels,
        out_indices=voxel_out_indices,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    occ_encoder_neck=dict(
        type='FPN3D',
        with_cp=True,
        in_channels=voxel_channels,
        out_channels=voxel_out_channel,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    pts_bbox_head=dict(
        type='OccHead',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        soft_weights=True,
        cascade_ratio=cascade_ratio,
        sample_from_voxel=sample_from_voxel,
        sample_from_img=sample_from_img,
        final_occ_size=occ_size,
        fine_topk=15000,
        empty_idx=empty_idx,
        num_level=len(voxel_out_indices),
        in_channels=[voxel_out_channel] * len(voxel_out_indices),
        out_channel=num_cls,
        point_cloud_range=point_cloud_range,
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0,
            loss_voxel_lovasz_weight=1.0,
        ),
         balance_cls_weight=True,
         class_num = num_cls,
         class_names = class_names        
    ),
             ###################Attention configs#######################
    cross_transformer=dict(
           type='PerceptionTransformer3D',
           rotate_prev_bev=True,
           use_shift=True,
           embed_dims=_dim_,
           num_cams = _num_cams_,
           encoder=dict(
               type='VoxFormerEncoder3D',
               num_layers=2,
               pc_range=point_cloud_range,
               num_points_in_pillar=10,
               return_intermediate=False,
               transformerlayers=dict(
                   type='VoxFormerLayer3D',
                   attn_cfgs=[
                       dict(
                           type='DeformCrossAttention3DCustom',
                           embed_dims=_dim_,
                           num_levels=1,
                           num_points=8)
                   ],
                   ffn_cfgs=dict(
                       type='FFN',
                       embed_dims=_dim_,
                       feedforward_channels=1024,
                       num_fcs=2,
                       ffn_drop=0.,
                       act_cfg=dict(type='ReLU', inplace=True),
                   ),
                   feedforward_channels=_ffn_dim_,
                   ffn_dropout=0.1,
                   operation_order=('self_attn', 'norm', 'ffn', 'norm')))
                   ),


    self_transformer=dict(
           type='PerceptionTransformer3D',
           rotate_prev_bev=True,
           use_shift=True,
           embed_dims=_dim_,
           num_cams = _num_cams_,
           encoder=dict(
               type='VoxFormerEncoder3D',
               num_layers=_num_layers_self_,
               pc_range=point_cloud_range,
               num_points_in_pillar=10,
               return_intermediate=False,
               transformerlayers=dict(
                   type='VoxFormerLayer3D',
                   attn_cfgs=[
                       dict(
                           type='DeformSelfAttention3DCustom',
                           embed_dims=_dim_,
                           num_levels=1,
                           num_points=_num_points_self_)
                   ],
                   ffn_cfgs=dict(
                       type='FFN',
                       embed_dims=_dim_,
                       feedforward_channels=1024,
                       num_fcs=2,
                       ffn_drop=0.,
                       act_cfg=dict(type='ReLU', inplace=True),
                   ),
                   feedforward_channels=_ffn_dim_,
                   ffn_dropout=0.1,
                   operation_order=('self_attn', 'norm', 'ffn', 'norm')))
                   ),
     positional_encoding=dict(
           type='LearnedPositionalEncoding3D',
           num_feats=_pos_dim_,
                 row_num_embed=128,
                 col_num_embed=128,
                 z_num_embed=20,
           ),
           
    empty_idx=empty_idx,
)



train_pipeline = [
    dict(type='LoadSparseRadar', to_float32=True
        ),
    dict(type='LoadOccupancy', to_float32=True, use_semantic=True, occ_path=occ_path, grid_size=occ_size, use_vel=False,
            unoccupied=empty_idx, pc_range=point_cloud_range, cal_visible=visible_mask),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_occ', 'sparse_radar']),
]

test_pipeline = [
    dict(type='LoadSparseRadar', to_float32=True
        ),
    dict(type='LoadOccupancy', to_float32=True, use_semantic=True, occ_path=occ_path, grid_size=occ_size, use_vel=False,
        unoccupied=empty_idx, pc_range=point_cloud_range, cal_visible=visible_mask),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names, with_label=False), 
    dict(type='Collect3D', keys=['gt_occ', 'sparse_radar'],
            meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token','radar_path','occ_path']),
]


test_config=dict(
    type=dataset_type,
    occ_root=occ_path,
    data_root=data_root,
    ann_file=val_ann_file,
    pipeline=test_pipeline,
    classes=class_names,
    modality=input_modality,
    occ_size=occ_size,
    pc_range=point_cloud_range,
)

train_config=dict(
        type=dataset_type,
        data_root=data_root,
        occ_root=occ_path,
        ann_file=train_ann_file,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        box_type_3d='LiDAR'),

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=train_config,
    val=test_config,
    test=test_config,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

optimizer = dict(
    type='AdamW',
    lr=3e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

runner = dict(type='EpochBasedRunner', max_epochs=15)
evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    save_best='SSC_mean',
    rule='greater',
)

custom_hooks = [
    dict(type='OccEfficiencyHook'),
]

