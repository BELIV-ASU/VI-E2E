import copy
from math import pi, cos, sin

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mmdet.models import HEADS, build_loss 
from mmdet.models.dense_heads import DETRHead
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.core import build_assigner, build_sampler
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from projects.mmdet3d_plugin.VAD.utils.traj_lr_warmup import get_traj_warmup_loss_weight
from projects.mmdet3d_plugin.VAD.utils.map_utils import (
    normalize_2d_pts, normalize_2d_bbox, denormalize_2d_pts, denormalize_2d_bbox
)
import json
from scipy.spatial.transform import Rotation as R
import os
import math
import random
from scipy.optimize import linear_sum_assignment
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion


#from projects.mmdet3d_plugin.VAD.v2xfusion_transformer import V2XTransformerDecoder


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_unit, verbose=False):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x

class LaneNet(nn.Module):
    def __init__(self, in_channels, hidden_unit, num_subgraph_layers):
        super(LaneNet, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(
                f'lmlp_{i}', MLP(in_channels, hidden_unit))
            in_channels = hidden_unit*2

    def forward(self, pts_lane_feats):
        '''
            Extract lane_feature from vectorized lane representation

        Args:
            pts_lane_feats: [batch size, max_pnum, pts, D]

        Returns:
            inst_lane_feats: [batch size, max_pnum, D]
        '''
        x = pts_lane_feats
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                # x [bs,max_lane_num,9,dim]
                x = layer(x)
                x_max = torch.max(x, -2)[0]
                x_max = x_max.unsqueeze(2).repeat(1, 1, x.shape[2], 1)
                x = torch.cat([x, x_max], dim=-1)
        x_max = torch.max(x, -2)[0]
        return x_max


@HEADS.register_module()
class VADHead(DETRHead):
    """Head of VAD model.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """
    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 fut_ts=6,
                 fut_mode=6,
                 loss_traj=dict(type='L1Loss', loss_weight=0.25),
                 loss_traj_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=0.8),
                 map_bbox_coder=None,
                 map_num_query=900,
                 map_num_classes=3,
                 map_num_vec=20,
                 map_num_pts_per_vec=2,
                 map_num_pts_per_gt_vec=2,
                 map_query_embed_type='all_pts',
                 map_transform_method='minmax',
                 map_gt_shift_pts_pattern='v0',
                 map_dir_interval=1,
                 map_code_size=None,
                 map_code_weights=None,
                 loss_map_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_map_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_map_iou=dict(type='GIoULoss', loss_weight=2.0),
                 loss_map_pts=dict(
                    type='ChamferDistance',loss_src_weight=1.0,loss_dst_weight=1.0
                 ),
                 loss_map_dir=dict(type='PtsDirCosLoss', loss_weight=2.0),
                 loss_map_curvature=dict(type='PtsCurvatureLoss', loss_weight=0.0),
                 tot_epoch=None,
                 use_traj_lr_warmup=False,
                 motion_decoder=None,
                 motion_map_decoder=None,
                 use_pe=False,
                 motion_det_score=None,
                 map_thresh=0.5,
                 dis_thresh=0.2,
                 pe_normalization=True,
                 ego_his_encoder=None,
                 ego_fut_mode=3,
                 loss_plan_reg=dict(type='L1Loss', loss_weight=0.25),
                 loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=0.1),
                 loss_plan_col=dict(type='PlanAgentDisLoss', loss_weight=0.1),
                 loss_plan_dir=dict(type='PlanMapThetaLoss', loss_weight=0.1),
                 ego_agent_decoder=None,
                 ego_map_decoder=None,
                 query_thresh=None,
                 query_use_fix_pad=None,
                 ego_lcf_feat_idx=None,
                 valid_fut_ts=6,
                 vi_agent_fuser=None,
                 vi_map_fuser=None,
                 vi_motion=None,
                 vi_map_motion=None,
                 agent_fusion_decoder=None,
                 agent_decoder = None,
                 agent_map_decoder = None,
                 map_agent_decoder = None,
                 v2x_use_time_compensation=True,
                 v2x_time_sync_max_dt=0.10,
                 v2x_time_sync_min_speed=0.05,
                 v2x_time_sync_max_speed=6.0,
                 v2x_time_sync_detach=True,
                 v2x_time_sync_start_epoch=0,
                 v2x_time_sync_score_thresh=0.15,
                 # Matching / fusion parameters (previously fell into **kwargs silently)
                 v2x_match_dist_thresh=2.0,
                 v2x_match_cls_cost=0.20,
                 v2x_match_score_cost=0.10,
                 v2x_anchor_fusion=True,
                 v2x_infra_conf_thresh=0.3,
                 v2x_min_inf_in_range=3,
                 v2x_spatial_overlap_thresh=15.0,
                 v2x_spatial_inject_thresh=15.0,
                 use_infra_map=False,
                 **kwargs):

        self.use_infra_map = use_infra_map
        self.vi_agent_fuser = vi_agent_fuser
        self.vi_map_fuser = vi_map_fuser
        self.vi_motion = vi_motion
        self.vi_map_motion = vi_map_motion
        self.agent_fusion_decoder = agent_fusion_decoder
        

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.tot_epoch = tot_epoch
        self.use_traj_lr_warmup = use_traj_lr_warmup
        self.motion_decoder = motion_decoder
        self.motion_map_decoder = motion_map_decoder
        self.use_pe = use_pe
        self.motion_det_score = motion_det_score
        self.map_thresh = map_thresh
        self.dis_thresh = dis_thresh
        self.pe_normalization = pe_normalization
        self.ego_his_encoder = ego_his_encoder
        self.ego_fut_mode = ego_fut_mode
        self.ego_agent_decoder = ego_agent_decoder
        self.ego_map_decoder = ego_map_decoder
        self.agent_decoder = agent_decoder
        self.agent_map_decoder = agent_map_decoder
        self.map_agent_decoder = map_agent_decoder
        self.query_thresh = query_thresh
        self.query_use_fix_pad = query_use_fix_pad
        self.ego_lcf_feat_idx = ego_lcf_feat_idx
        self.valid_fut_ts = valid_fut_ts
        self.v2x_use_time_compensation = bool(v2x_use_time_compensation)
        self.v2x_time_sync_max_dt = float(v2x_time_sync_max_dt)
        self.v2x_time_sync_min_speed = float(v2x_time_sync_min_speed)
        self.v2x_time_sync_max_speed = float(v2x_time_sync_max_speed)
        self.v2x_time_sync_detach = bool(v2x_time_sync_detach)
        self.v2x_time_sync_start_epoch = int(v2x_time_sync_start_epoch)
        self.v2x_time_sync_score_thresh = float(v2x_time_sync_score_thresh)
        self.v2x_match_dist_thresh = float(v2x_match_dist_thresh)
        self.v2x_match_cls_cost = float(v2x_match_cls_cost)
        self.v2x_match_score_cost = float(v2x_match_score_cost)
        self.v2x_anchor_fusion = bool(v2x_anchor_fusion)
        self.v2x_infra_conf_thresh = float(v2x_infra_conf_thresh)
        self.v2x_min_inf_in_range = int(v2x_min_inf_in_range)
        self.v2x_spatial_overlap_thresh = float(v2x_spatial_overlap_thresh)
        self.v2x_spatial_inject_thresh = float(v2x_spatial_inject_thresh)

        if loss_traj_cls['use_sigmoid'] == True:
            self.traj_num_cls = 1
        else:
          self.traj_num_cls = 2

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        if map_code_size is not None:
            self.map_code_size = map_code_size
        else:
            self.map_code_size = 10
        if map_code_weights is not None:
            self.map_code_weights = map_code_weights
        else:
            self.map_code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1

        self.map_bbox_coder = build_bbox_coder(map_bbox_coder)
        self.map_query_embed_type = map_query_embed_type
        self.map_transform_method = map_transform_method
        self.map_gt_shift_pts_pattern = map_gt_shift_pts_pattern
        map_num_query = map_num_vec * map_num_pts_per_vec
        self.map_num_query = map_num_query
        self.map_num_classes = map_num_classes
        self.map_num_vec = map_num_vec
        self.map_num_pts_per_vec = map_num_pts_per_vec
        self.map_num_pts_per_gt_vec = map_num_pts_per_gt_vec
        self.map_dir_interval = map_dir_interval

        if loss_map_cls['use_sigmoid'] == True:
            self.map_cls_out_channels = map_num_classes
        else:
            self.map_cls_out_channels = map_num_classes + 1

        self.map_bg_cls_weight = 0
        map_class_weight = loss_map_cls.get('class_weight', None)
        if map_class_weight is not None and (self.__class__ is VADHead):
            assert isinstance(map_class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(map_class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            map_bg_cls_weight = loss_map_cls.get('bg_cls_weight', map_class_weight)
            assert isinstance(map_bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(map_bg_cls_weight)}.'
            map_class_weight = torch.ones(map_num_classes + 1) * map_class_weight
            # set background class as the last indice
            map_class_weight[map_num_classes] = map_bg_cls_weight
            loss_map_cls.update({'class_weight': map_class_weight})
            if 'bg_cls_weight' in loss_map_cls:
                loss_map_cls.pop('bg_cls_weight')
            self.map_bg_cls_weight = map_bg_cls_weight
        
        self.traj_bg_cls_weight = 0

        # The DETRHead parent asserts loss_cls.weight == assigner.cls_cost.weight.
        # When detection losses are intentionally disabled (loss_cls=0.0) but we
        # still want meaningful assigner costs for trajectory target matching, we
        # temporarily zero the detection assigner costs before calling super().__init__,
        # then rebuild self.assigner with the real costs afterward.
        _train_cfg = kwargs.get('train_cfg')
        _real_det_assigner_cfg = None
        if _train_cfg is not None and kwargs.get('loss_cls', {}).get('loss_weight', 1.0) == 0.0:
            import copy as _copy
            _real_det_assigner_cfg = _copy.deepcopy(_train_cfg.get('assigner', {}))
            _zeroed = _copy.deepcopy(_train_cfg['assigner'])
            for cost_key in ('cls_cost', 'reg_cost', 'iou_cost'):
                if cost_key in _zeroed:
                    _zeroed[cost_key]['weight'] = 0.0
            _train_cfg['assigner'] = _zeroed

        super(VADHead, self).__init__(*args, transformer=transformer, **kwargs)

        # Restore the real assigner with meaningful costs for trajectory matching.
        if _real_det_assigner_cfg:
            from mmdet.core import build_assigner as _build_assigner
            self.assigner = _build_assigner(_real_det_assigner_cfg)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.map_code_weights = nn.Parameter(torch.tensor(
            self.map_code_weights, requires_grad=False), requires_grad=False)
        
        if kwargs['train_cfg'] is not None:
            assert 'map_assigner' in kwargs['train_cfg'], 'map assigner should be provided '\
                'when train_cfg is set.'
            map_assigner = kwargs['train_cfg']['map_assigner']
            assert loss_map_cls['loss_weight'] == map_assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_map_bbox['loss_weight'] == map_assigner['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'
            assert loss_map_iou['loss_weight'] == map_assigner['iou_cost']['weight'], \
                'The regression iou weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_map_pts['loss_weight'] == map_assigner['pts_cost']['weight'], \
                'The regression l1 weight for map pts loss and matcher should be' \
                'exactly the same.'

            self.map_assigner = build_assigner(map_assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.map_sampler = build_sampler(sampler_cfg, context=self)
        
        self.loss_traj = build_loss(loss_traj)
        self.loss_traj_cls = build_loss(loss_traj_cls)
        self.loss_map_bbox = build_loss(loss_map_bbox)
        self.loss_map_cls = build_loss(loss_map_cls)
        self.loss_map_iou = build_loss(loss_map_iou)
        self.loss_map_pts = build_loss(loss_map_pts)
        self.loss_map_dir = build_loss(loss_map_dir)
        self.loss_map_curvature = build_loss(loss_map_curvature)
        self.loss_plan_reg = build_loss(loss_plan_reg)
        self.loss_plan_bound = build_loss(loss_plan_bound)
        self.loss_plan_col = build_loss(loss_plan_col)
        self.loss_plan_dir = build_loss(loss_plan_dir)
        
        with open('/home/jingxiong/V2X-Seq-SPD/infrastructure-side/v1.0-trainval/ego_pose.json','r') as f:
            self.ego_pose_infra = json.load(f)
        
        with open('/home/jingxiong/V2X-Seq-SPD/vehicle-side/v1.0-trainval/ego_pose.json','r') as f:
            self.ego_pose_vehicle = json.load(f)

        self.map_transform_mlp = nn.Linear(16, self.embed_dims) 
        self.agent_transform_mlp = nn.Linear(16, self.embed_dims)
        self.norm_map_pos = nn.LayerNorm(self.embed_dims)
        self.norm_agent_pos = nn.LayerNorm(self.embed_dims)
        self.bev_embed_linear = nn.Linear(self.embed_dims, self.embed_dims)
        self.bev_pos_linear = nn.Linear(self.embed_dims, self.embed_dims)
        self.cross_agent_fusion = nn.Linear(self.embed_dims, self.embed_dims)


        self.frameid_to_ts = {}
        with open('/home/jingxiong/V2X-Seq-SPD/vehicle-side/data_info.json', 'r') as f:
            data_info = json.load(f)
        self.frameid_to_ts["ego"] = {entry['frame_id']: int(entry['image_timestamp']) for entry in data_info}

        with open('/home/jingxiong/V2X-Seq-SPD/infrastructure-side/data_info.json', 'r') as f:
            data_info = json.load(f)
        self.frameid_to_ts["infra"] = {entry['frame_id']: int(entry['image_timestamp']) for entry in data_info}



    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        cls_branch = nn.Sequential(*cls_branch)

        cls_branch_fuse = []
        for _ in range(self.num_reg_fcs):
            cls_branch_fuse.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch_fuse.append(nn.LayerNorm(self.embed_dims))
            cls_branch_fuse.append(nn.ReLU(inplace=True))
        cls_branch_fuse.append(Linear(self.embed_dims, self.cls_out_channels))
        cls_branch_fuse = nn.Sequential(*cls_branch_fuse)

        

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        reg_branch_fuse = []
        for _ in range(self.num_reg_fcs):
            reg_branch_fuse.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch_fuse.append(nn.ReLU(inplace=True))
        reg_branch_fuse.append(Linear(self.embed_dims, self.code_size))
        reg_branch_fuse = nn.Sequential(*reg_branch_fuse)

        

        traj_branch = []
        for _ in range(self.num_reg_fcs):
            traj_branch.append(Linear(self.embed_dims*2, self.embed_dims*2))
            traj_branch.append(nn.ReLU())
        traj_branch.append(Linear(self.embed_dims*2, self.fut_ts*2))
        traj_branch = nn.Sequential(*traj_branch)

        traj_cls_branch = []
        for _ in range(self.num_reg_fcs):
            traj_cls_branch.append(Linear(self.embed_dims*2, self.embed_dims*2))
            traj_cls_branch.append(nn.LayerNorm(self.embed_dims*2))
            traj_cls_branch.append(nn.ReLU(inplace=True))
        traj_cls_branch.append(Linear(self.embed_dims*2, self.traj_num_cls))
        traj_cls_branch = nn.Sequential(*traj_cls_branch)

        map_cls_branch = []
        for _ in range(self.num_reg_fcs):
            map_cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            map_cls_branch.append(nn.LayerNorm(self.embed_dims))
            map_cls_branch.append(nn.ReLU(inplace=True))
        map_cls_branch.append(Linear(self.embed_dims, self.map_cls_out_channels))
        map_cls_branch = nn.Sequential(*map_cls_branch)

        map_cls_branch_fuse = []
        for _ in range(self.num_reg_fcs):
            map_cls_branch_fuse.append(Linear(self.embed_dims, self.embed_dims))
            map_cls_branch_fuse.append(nn.LayerNorm(self.embed_dims))
            map_cls_branch_fuse.append(nn.ReLU(inplace=True))
        map_cls_branch_fuse.append(Linear(self.embed_dims, self.map_cls_out_channels))
        map_cls_branch_fuse = nn.Sequential(*map_cls_branch_fuse)

        map_reg_branch = []
        for _ in range(self.num_reg_fcs):
            map_reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            map_reg_branch.append(nn.ReLU())
        map_reg_branch.append(Linear(self.embed_dims, self.map_code_size))
        map_reg_branch = nn.Sequential(*map_reg_branch)

        map_reg_branch_fuse = []
        for _ in range(self.num_reg_fcs):
            map_reg_branch_fuse.append(Linear(self.embed_dims, self.embed_dims))
            map_reg_branch_fuse.append(nn.ReLU(inplace=True))
        map_reg_branch_fuse.append(Linear(self.embed_dims, self.code_size))
        map_reg_branch_fuse = nn.Sequential(*map_reg_branch_fuse)


        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_decoder_layers = 1
        num_map_decoder_layers = 1
        if self.transformer.decoder is not None:
            num_decoder_layers = self.transformer.decoder.num_layers
        if self.transformer.map_decoder is not None:
            num_map_decoder_layers = self.transformer.map_decoder.num_layers
        num_motion_decoder_layers = 1
        num_pred = (num_decoder_layers + 1) if \
            self.as_two_stage else num_decoder_layers
        motion_num_pred = (num_motion_decoder_layers + 1) if \
            self.as_two_stage else num_motion_decoder_layers
        map_num_pred = (num_map_decoder_layers + 1) if \
            self.as_two_stage else num_map_decoder_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(cls_branch, num_pred)
            self.cls_branches_fuse = _get_clones(cls_branch_fuse, num_pred)
            
            self.reg_branches = _get_clones(reg_branch, num_pred)
            self.reg_branches_fuse = _get_clones(reg_branch_fuse, num_pred)
            
            self.traj_branches = _get_clones(traj_branch, motion_num_pred)
            self.traj_cls_branches = _get_clones(traj_cls_branch, motion_num_pred)
            self.map_cls_branches = _get_clones(map_cls_branch, map_num_pred)
            self.map_cls_branches_fuse = _get_clones(map_cls_branch_fuse, map_num_pred)
            self.map_reg_branches = _get_clones(map_reg_branch, map_num_pred)
            self.map_reg_branches_fuse = _get_clones(map_reg_branch_fuse, map_num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [cls_branch for _ in range(num_pred)])
            self.cls_branches_fuse = nn.ModuleList(
                [cls_branch_fuse for _ in range(num_pred)])
            
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
            self.reg_branches_fuse = nn.ModuleList(
                [reg_branch_fuse for _ in range(num_pred)])
            
            
            self.traj_branches = nn.ModuleList(
                [traj_branch for _ in range(motion_num_pred)])
            self.traj_cls_branches = nn.ModuleList(
                [traj_cls_branch for _ in range(motion_num_pred)])
            
            self.map_cls_branches = nn.ModuleList(
                [map_cls_branch for _ in range(map_num_pred)])
            self.map_cls_branches_fuse = nn.ModuleList(
                [map_cls_branch_fuse for _ in range(map_num_pred)])
            self.map_reg_branches = nn.ModuleList(
                [map_reg_branch for _ in range(map_num_pred)])
            self.map_reg_branches_fuse = nn.ModuleList(
                [map_reg_branch_fuse for _ in range(map_num_pred)])

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)
            if self.map_query_embed_type == 'all_pts':
                self.map_query_embedding = nn.Embedding(self.map_num_query,
                                                    self.embed_dims * 2)
            elif self.map_query_embed_type == 'instance_pts':
                self.map_query_embedding = None
                self.map_instance_embedding = nn.Embedding(self.map_num_vec, self.embed_dims * 2)
                self.map_pts_embedding = nn.Embedding(self.map_num_pts_per_vec, self.embed_dims * 2)
        
        if self.motion_decoder is not None:
            self.motion_decoder = build_transformer_layer_sequence(self.motion_decoder)
            self.motion_mode_query = nn.Embedding(self.fut_mode, self.embed_dims)	
            self.motion_mode_query.weight.requires_grad = True
            if self.use_pe:
                self.pos_mlp_sa = nn.Linear(2, self.embed_dims)
        else:
            raise NotImplementedError('Not implement yet')

        if self.motion_map_decoder is not None:
            self.lane_encoder = LaneNet(256, 128, 3)
            self.motion_map_decoder = build_transformer_layer_sequence(self.motion_map_decoder)
            if self.use_pe:
                self.pos_mlp = nn.Linear(2, self.embed_dims)
        
        if self.ego_his_encoder is not None:
            self.ego_his_encoder = LaneNet(2, self.embed_dims//2, 3)
        else:
            self.ego_query = nn.Embedding(1, self.embed_dims)	

        if self.ego_agent_decoder is not None:
            self.ego_agent_decoder = build_transformer_layer_sequence(self.ego_agent_decoder)
            if self.use_pe:
                self.ego_agent_pos_mlp = nn.Linear(2, self.embed_dims)

        if self.ego_map_decoder is not None:
            self.ego_map_decoder = build_transformer_layer_sequence(self.ego_map_decoder)
            if self.use_pe:
                self.ego_map_pos_mlp = nn.Linear(2, self.embed_dims)

        if self.agent_decoder is not None:
            self.agent_decoder = build_transformer_layer_sequence(self.agent_decoder)

        if self.agent_map_decoder is not None:
            self.agent_map_decoder = build_transformer_layer_sequence(self.agent_map_decoder)
        
        if self.map_agent_decoder is not None:
            self.map_agent_decoder = build_transformer_layer_sequence(self.map_agent_decoder)
        ego_fut_decoder = []
        ego_fut_dec_in_dim = self.embed_dims*2 + len(self.ego_lcf_feat_idx) \
            if self.ego_lcf_feat_idx is not None else self.embed_dims*2
        for _ in range(self.num_reg_fcs):
            ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, ego_fut_dec_in_dim))
            ego_fut_decoder.append(nn.LayerNorm(ego_fut_dec_in_dim, eps=1e-6))
            ego_fut_decoder.append(nn.ReLU())
        ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, self.ego_fut_mode*self.fut_ts*2))
        self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)

        self.agent_fus_mlp = nn.Sequential(
            nn.Linear(self.fut_mode*2*self.embed_dims, self.embed_dims, bias=True),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims, bias=True))
        
        # Modification:
        if self.vi_agent_fuser is not None: 
            self.vi_agent_fuser = build_transformer_layer_sequence(self.vi_agent_fuser)
        
        if self.vi_map_fuser is not None:
            self.vi_map_fuser = build_transformer_layer_sequence(self.vi_map_fuser)
        
        if self.vi_motion is not None:
            self.vi_motion = build_transformer_layer_sequence(self.vi_motion)

        if self.vi_map_motion is not None:
            self.vi_map_motion = build_transformer_layer_sequence(self.vi_map_motion)

        if self.agent_fusion_decoder is not None:
            self.agent_fusion_decoder = build_transformer_layer_sequence(self.agent_fusion_decoder)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
            for m in self.cls_branches_fuse:
                nn.init.constant_(m[-1].bias, bias_init)
        if self.loss_map_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.map_cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        if self.loss_traj_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.traj_cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        # for m in self.map_reg_branches:
        #     constant_init(m[-1], 0, bias=0)
        # nn.init.constant_(self.map_reg_branches[0][-1].bias.data[2:], 0.)
        if self.motion_decoder is not None:
            for p in self.motion_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            nn.init.orthogonal_(self.motion_mode_query.weight)
            if self.use_pe:
                xavier_init(self.pos_mlp_sa, distribution='uniform', bias=0.)
        if self.motion_map_decoder is not None:
            for p in self.motion_map_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            for p in self.lane_encoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            if self.use_pe:
                xavier_init(self.pos_mlp, distribution='uniform', bias=0.)
        if self.ego_his_encoder is not None:
            for p in self.ego_his_encoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        if self.ego_agent_decoder is not None:
            for p in self.ego_agent_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        if self.ego_map_decoder is not None:
            for p in self.ego_map_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.ego_fut_decoder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def get_timestamp_from_frame_id(self, frame_id, mode):
        try:
            return self.frameid_to_ts[mode][frame_id]
        except KeyError:
            raise ValueError(f"Timestamp for frame_id {frame_id} not found.")


    # Modifications: Generate the map, agent, ego queries
    def query_generation(self, outputs):
        bev_embed, hs, init_reference, inter_references, \
            map_hs, map_init_reference, map_inter_references = outputs

        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        outputs_coords_bev = []
        

        map_hs = map_hs.permute(0, 2, 1, 3) #check whether it's BEV by visualizing
        map_outputs_classes = []
        map_outputs_coords = []
        map_outputs_pts_coords = []
        map_outputs_coords_bev = []

        #Agent:
        for lvl in range(hs.shape[0]): # used to get agent coordinates and their confidence
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3


            tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            outputs_coords_bev.append(tmp[..., 0:2].clone().detach())
            tmp[..., 4:5] = tmp[..., 4:5] + reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            
            
            veh_tmp = tmp.clone()
            veh_class = outputs_class.clone()
            outputs_coords.append(veh_tmp)
            

            outputs_classes.append(outputs_class)
            
        # Map elements:
        for lvl in range(map_hs.shape[0]):
            if lvl == 0:
                reference = map_init_reference
            else:
                reference = map_inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            map_outputs_class = self.map_cls_branches[lvl](
                map_hs[lvl].view(map_hs[lvl].shape[0], self.map_num_vec, self.map_num_pts_per_vec, -1).mean(2)
            )
            tmp = self.map_reg_branches[lvl](map_hs[lvl])
            # TODO: check the shape of reference
            assert reference.shape[-1] == 2
            #tmp = tmp.clone()
            #tmp[..., 0:2] = tmp[..., 0:2]+reference[..., 0:2]
            tmp_1 = tmp[..., 0:2]+reference[..., 0:2]
            tmp[..., 0:2] = tmp_1
            tmp_2 = tmp.clone()
            tmp = tmp_2.sigmoid() # cx,cy,w,h
            map_outputs_coord, map_outputs_pts_coord = self.map_transform_box(tmp)
            map_outputs_coords_bev.append(map_outputs_pts_coord.clone().detach())
            map_outputs_classes.append(map_outputs_class) #map class
            map_outputs_coords.append(map_outputs_coord) #map class coordinates
            map_outputs_pts_coords.append(map_outputs_pts_coord)
        
        # Vectorized Motion Transformer
        # self.motion_decoder: get motion queries (self-attention) 
        batch_size, num_agent = outputs_coords_bev[-1].shape[:2]
        agent_query = hs[-1].permute(1, 0, 2)  # [A, B, D]

        map_query = map_hs[-1].view(batch_size, self.map_num_vec, self.map_num_pts_per_vec, -1)
        map_query = self.lane_encoder(map_query)  # [B, P, pts, D] -> [B, P, D]
        map_conf = map_outputs_classes[-1] #[P_C1,PC2,PC3]
        map_pos = map_outputs_coords_bev[-1]
        map_pos_raw = map_outputs_coords_bev[-1]
        # use the most close pts pos in each map inst as the inst's pos
        
        batch, num_map = map_pos.shape[:2]
        map_dis = torch.sqrt(map_pos[..., 0]**2 + map_pos[..., 1]**2)
        min_map_pos_idx = map_dis.argmin(dim=-1).flatten()  # [B*P]
        min_map_pos = map_pos.flatten(0, 1)  # [B*P, pts, 2]
        min_map_pos = min_map_pos[range(min_map_pos.shape[0]), min_map_pos_idx]  # [B*P, 2]
        min_map_pos = min_map_pos.view(batch, num_map, 2)  # [B, P, 2]

        # map_pos = min_map_pos

        map_query, map_pos, map_mask = self.select_and_pad_query(
            map_query, min_map_pos, map_conf,
            score_thresh=self.query_thresh, use_fix_pad=self.query_use_fix_pad
        )
        map_pos_emb = self.ego_map_pos_mlp(map_pos)

        if self.use_pe:
            map_pos = self.pos_mlp(map_pos)
        else:
            map_pos = None
        
        # planning: get ego queries
        (num_agent, batch) = agent_query.shape[:2]#motion_hs.shape[:2] (batch, num_agent)
        ego_his_feats = self.ego_query.weight.unsqueeze(0).repeat(batch, 1, 1)

        #Ego query:
        ego_query = ego_his_feats
        ego_pos = torch.zeros((batch, 1, 2), device=ego_query.device)
        ego_pos_emb = self.ego_agent_pos_mlp(ego_pos)

        #Mask/Filter less confident agent queries
        agent_conf = outputs_classes[-1]
        agent_pos = outputs_coords_bev[-1]
        agent_query = agent_query.permute(1,0,2) 
        
        agent_query, agent_pos, agent_mask = self.select_and_pad_query(
            agent_query, agent_pos, agent_conf,
            score_thresh=self.query_thresh, use_fix_pad=self.query_use_fix_pad
        )
        
        agent_mask = agent_mask.transpose(0, 1)
        agent_pos_emb = self.ego_agent_pos_mlp(agent_pos)
        
        ego_pos = torch.zeros((batch, 1, 2), device=agent_query.device)
        ego_pos_emb = self.ego_map_pos_mlp(ego_pos)

        return ego_query, agent_query, ego_pos_emb, agent_pos_emb, agent_mask, map_query, map_pos_emb, map_mask,\
        ego_his_feats, veh_tmp, veh_class, outputs_coords, outputs_classes, outputs_coords_bev, map_outputs_coords, map_outputs_classes, map_outputs_coords_bev,map_outputs_pts_coords

    def get_transform(self,ego_pose, token):
        for entry in ego_pose:
            if entry["token"] == token:
                rot = Quaternion(entry["rotation"]).rotation_matrix#R.from_quat(entry["rotation"]).as_matrix()  # [3, 3]
                trans = np.array(entry["translation"])  # [3]
                T = np.eye(4)
                T[:3, :3] = rot
                T[:3, 3] = trans
                return T
        raise ValueError(f"Token {token} not found")

    def apply_transform(self,points, T):
        # Ensure T is a torch tensor if it's not already
        if not isinstance(T, torch.Tensor):
            # Assuming T is a NumPy array from get_transform, convert it to a torch tensor
            # and move it to the device of 'points'
            T = torch.from_numpy(T).to(points.device).to(points.dtype)

        # Check if points has a batch dimension
        has_batch_dim = points.dim() == 3
        if has_batch_dim:
            B, N, D_in = points.shape # D_in is the original dimension of points (e.g., 2 or 3)
            points_reshaped = points.view(B * N, D_in)
        else:
            N, D_in = points.shape # D_in is the original dimension of points (e.g., 2 or 3)
            points_reshaped = points # Already [N, D_in]

        # Handle points with 2 dimensions by adding a Z-coordinate of 0
        if D_in == 2:
            # Pad with a zero Z-coordinate: [N, 2] -> [N, 3] (x, y, 0)
            zeros = torch.zeros((points_reshaped.shape[0], 1), device=points.device, dtype=points.dtype)
            points_3d = torch.cat([points_reshaped, zeros], dim=1) # [N, 3]
        elif D_in == 3:
            points_3d = points_reshaped # Already [N, 3]
        else:
            raise ValueError(f"Unsupported point dimension: {D_in}. Expected 2 or 3.")

        # Create homogeneous coordinates [N, 4]
        ones = torch.ones((points_3d.shape[0], 1), device=points.device, dtype=points.dtype)
        points_h = torch.cat([points_3d, ones], dim=1) # [N, 4]

        # Apply transform: (T @ points_h.T).T
        # T is [4, 4], points_h is [N, 4]
        # (T @ points_h.T) will be [4, N]
        transformed_points_h = T @ points_h.T # [4, N]
        
        # Extract the transformed 3D points [N, 3]
        transformed_points_3d = transformed_points_h[:3, :].T # [N, 3]

        # If original points were 2D, return only the transformed X, Y
        if D_in == 2:
            transformed_points = transformed_points_3d[:, :2] # [N, 2]
        else: # D_in == 3
            transformed_points = transformed_points_3d # [N, 3]

        if has_batch_dim:
            # Reshape back to [B, N, D_in]
            return transformed_points.view(B, N, D_in)
        else:
            return transformed_points

    def get_time_delay(self, img_metas):
        delta_t_gt = []
        for meta in img_metas:
            filenames = meta.get('filename', [])
            if len(filenames) < 2:
                delta_t_gt.append([0.0])
                continue
            ego_frame_id = os.path.splitext(os.path.basename(filenames[0]))[0]
            infra_frame_id = os.path.splitext(os.path.basename(filenames[1]))[0]
            try:
                ts_ego = self.get_timestamp_from_frame_id(ego_frame_id, mode="ego")
                ts_infra = self.get_timestamp_from_frame_id(infra_frame_id, mode="infra")
            except ValueError as e:
                delta_t_gt.append([0.0])
                continue
            dt_us = ts_infra - ts_ego
            # Signed offset in seconds: infra_ts - vehicle_ts.
            # Positive → infra image is later than vehicle image.
            delta_t_gt.append([dt_us])
        return torch.tensor(delta_t_gt).float() / 1e6

    def _get_coop_bev_embed(self, bev_embed_src, bev_pos_src, query, query_pos, reference_points, start_idx):
        bev_embed = bev_embed_src.clone()
        bev_pos = bev_pos_src.clone()  

        # print('act_track_instances len:',len(act_track_instances))

        locs = reference_points.squeeze(0)[start_idx:,:].clone()
        
        locs[:, 0:1] = locs[:, 0:1] * self.bev_w # w
        locs[:, 1:2] = locs[:, 1:2] * self.bev_h # h

        pixel_len = 2 # 2
        #import pdb; pdb.set_trace()

        for idx in range(locs.shape[0]):
            w = int(locs[idx, 0])
            h = int(locs[idx, 1])
            if w >= self.bev_w or w < 0 or h >= self.bev_h or h < 0:
                continue

            for hh in range(max(0, h - pixel_len), min(self.bev_h - 1, h + pixel_len)):
                for ww in range(max(0, w - pixel_len), min(self.bev_w - 1, w + pixel_len)):
                    bev_embed[hh * self.bev_w + ww, :, :] =  bev_embed[hh * self.bev_w + ww, :, :] + self.bev_embed_linear(query.squeeze(1)[start_idx:,:][idx, :])
                    bev_pos[:, :, hh, ww] = bev_pos[:, :, hh, ww] + self.bev_pos_linear(query_pos.squeeze(1)[start_idx:,:][idx, :])
                    
                    
        return bev_embed, bev_pos

    def transform_pc_range_aabb(self,pc_range, T_src_to_dst):
        """
        pc_range: [xmin, ymin, zmin, xmax, ymax, zmax]
        T_src_to_dst: 4x4 matrix, source frame -> destination frame
        returns axis-aligned [xmin, ymin, zmin, xmax, ymax, zmax] in destination frame
        """
        xmin, ymin, zmin, xmax, ymax, zmax = pc_range

        corners = np.array([
            [x, y, z, 1.0]
            for x in [xmin, xmax]
            for y in [ymin, ymax]
            for z in [zmin, zmax]
        ])

        corners_dst = corners @ T_src_to_dst.T
        mins = corners_dst[:, :3].min(axis=0)
        maxs = corners_dst[:, :3].max(axis=0)

        return [*mins.tolist(), *maxs.tolist()]

    # @auto_fp16(apply_to=('mlvl_feats'))
    @force_fp32(apply_to=('mlvl_feats', 'prev_bev'))
    def forward(self,
                mlvl_feats,
                img_metas,
                prev_bev=None,
                only_bev=False,
                ego_his_trajs=None,
                ego_lcf_feat=None,
                infra_queries=None,
                training_mode=None,
                infra_global_pc_range = None,
                gt_bboxes_3d = None,
            ):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)        
        if self.map_query_embed_type == 'all_pts':
            map_query_embeds = self.map_query_embedding.weight.to(dtype)
        elif self.map_query_embed_type == 'instance_pts':
            map_pts_embeds = self.map_pts_embedding.weight.unsqueeze(0)
            map_instance_embeds = self.map_instance_embedding.weight.unsqueeze(1)
            map_query_embeds = (map_pts_embeds + map_instance_embeds).flatten(0, 1).to(dtype)

        bev_queries = self.bev_embedding.weight.to(dtype)
        
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
            
        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            # Get BEV features in transformer
          
            vehicle_feats = mlvl_feats

            if training_mode:
                #with torch.no_grad():
                outputs = self.transformer(
                    vehicle_feats,
                    bev_queries,
                    object_query_embeds,
                    map_query_embeds,
                    self.bev_h,
                    self.bev_w,
                    grid_length=(self.real_h / self.bev_h,
                                 self.real_w / self.bev_w),
                    bev_pos=bev_pos,
                    reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                    cls_branches=self.cls_branches if self.as_two_stage else None,
                    map_reg_branches=self.map_reg_branches if self.with_box_refine else None,  # noqa:E501
                    map_cls_branches=self.map_cls_branches if self.as_two_stage else None,
                    img_metas=img_metas,
                    prev_bev=prev_bev
                )
            else:
                outputs = self.transformer(
                    vehicle_feats,
                    bev_queries,
                    object_query_embeds,
                    map_query_embeds,
                    self.bev_h,
                    self.bev_w,
                    grid_length=(self.real_h / self.bev_h,
                                 self.real_w / self.bev_w),
                    bev_pos=bev_pos,
                    reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                    cls_branches=self.cls_branches if self.as_two_stage else None,
                    map_reg_branches=self.map_reg_branches if self.with_box_refine else None,  # noqa:E501
                    map_cls_branches=self.map_cls_branches if self.as_two_stage else None,
                    img_metas=img_metas,
                    prev_bev=prev_bev
                )
        

        # bev_embed: [height*width, batch_size, dim] [10000, 1, 256]
        # hs: Agent state [layers, num, bs, dim] [3, 300, 1, 256]
        # init_reference: [1, 300, 3], inter_reference: [3, 1, 300, 3]
        # map_hs: Map state [layers, num, bs, dim] [3, 2000, 1, 256]
        # map_init_reference: [1, 2000, 2] map_inter_references: [3, 1, 2000, 2]
        # Prepare the BEV features, Map queries, Agent queries

        bev_embed, hs, init_reference, inter_references, \
            map_hs, map_init_reference, map_inter_references = outputs  # sigmoid coordinates
        
        #prepare transformation matrices from file
        sample_token_veh = img_metas[0]['sample_idx']

        lidar2ego_r = Quaternion([0.7088183911535113, 0.006425301516570595, -0.00037209309691732854, 0.7053616557551852]).rotation_matrix
        lidar2ego_t = np.array([-0.007779389591034794, 0.1976561805497153, 0.263])
        lidar2ego = np.eye(4)
        lidar2ego[:3,:3] = lidar2ego_r
        lidar2ego[:3,3] = lidar2ego_t

        ego2global = self.get_transform(self.ego_pose_vehicle, sample_token_veh) # ego2global

        #lidar2global = ego2global @ lidar2ego
        T_veh = ego2global @ lidar2ego # converts lidar points to global

        #get query from modality

        #ego
         
        ego_query, agent_query_v, ego_pos_emb, agent_pos_emb, agent_mask, map_query, map_pos_emb, map_mask,\
        ego_his_feats, veh_tmp, veh_class, veh_outputs_coords, veh_outputs_classes,veh_outputs_coords_bev, veh_map_outputs_coords, veh_map_outputs_classes, veh_map_outputs_coords_bev, veh_map_outputs_pts_coords = self.query_generation(outputs)

        if training_mode:
            #infra
            if len(infra_queries) >= 17:
                ego_query_i, agent_query_i, ego_pos_emb_i, agent_pos_emb_i, agent_mask_i, map_query_i, map_pos_emb_i, map_mask_i,\
                ego_his_feats_i, ca_motion_query_i, infra_global_coords, infra_classes, infra_coords, infra_tmp, infra_global_coords_list, infra_tmps, infra_global_vels_list = infra_queries[:17]
            else:
                ego_query_i, agent_query_i, ego_pos_emb_i, agent_pos_emb_i, agent_mask_i, map_query_i, map_pos_emb_i, map_mask_i,\
                ego_his_feats_i, ca_motion_query_i, infra_global_coords, infra_classes, infra_coords, infra_tmp, infra_global_coords_list, infra_tmps = infra_queries
                infra_global_vels_list = None

            # Compute infra→vehicle frame rotation for correcting heading and velocity.
            # The infra decoder outputs sin/cos/vx/vy in infra-local frame but the GT
            # used in the loss is in vehicle-lidar frame. Without this rotation the
            # supervision signal for heading and velocity is wrong regardless of training
            # duration, explaining why mean_speed stays at 1-2 m/s across epochs.
            _yaw_offset_inf_to_veh = 0.0  # fallback if token lookup fails
            _R_g2v_np = None
            try:
                _infra_token = img_metas[0]['filename'][1].split('/')[-1][:-4]
                _T_inf_np = self.get_transform(self.ego_pose_infra, _infra_token)
                self._diag_T_inf_cache = _T_inf_np  # expose for TimeComp rotation-comparison diag
                _T_veh_inv_np = np.linalg.inv(T_veh)
                _R_inf_to_veh_np = _T_veh_inv_np[:2, :2] @ _T_inf_np[:2, :2]
                _yaw_offset_inf_to_veh = float(
                    np.arctan2(_R_inf_to_veh_np[1, 0], _R_inf_to_veh_np[0, 0]))
                _R_g2v_np = _T_veh_inv_np[:2, :2]  # used to rotate global vel → vehicle
                if not getattr(self, '_yaw_offset_printed', False):
                    print(f'[HeadingFix] infra→vehicle yaw_offset={np.degrees(_yaw_offset_inf_to_veh):.1f}°  '
                          f'(non-zero means heading/vel supervision was wrong before this fix)')
                    self._yaw_offset_printed = True
            except Exception:
                pass

        #prepare ego position in global coordinate system
        ego_pos = torch.zeros((1, 1, 2), device=agent_query_v.device)
        ego_pos = self.apply_transform(ego_pos,T_veh)
        ego_pos_emb = self.ego_map_pos_mlp(ego_pos)


        #Spatial synchronization
        #vehicle global coordinate embedding

        T_veh_tensor = torch.tensor(T_veh, dtype=map_query.dtype, device=map_query.device).view(1, 1, -1)  # [1, 1, 16]
        T_veh_embed_map = self.map_transform_mlp(T_veh_tensor.expand(map_query.shape[0], map_query.shape[1], -1))  # [B, M, D]
        map_pos_emb = self.norm_map_pos(map_pos_emb.clone() + T_veh_embed_map) # should we remove normalization?
        #map_query = map_query.clone() + map_pos_emb
        veh_map_pos_embed = map_pos_emb

        #T_veh_embed_agent = self.agent_transform_mlp(T_veh_tensor.expand(agent_query_v.shape[0], agent_query_v.shape[1], -1))  # [B, N, D]
        #agent_pos_emb = self.norm_agent_pos(agent_pos_emb + T_veh_embed_agent)
        #agent_query = agent_query.clone() + agent_pos_emb
        #veh_agent_pos_embed = agent_pos_emb

        # Do late & intermediate fusion here
        if training_mode:
            # --------------------------- Temporal synchronization -----------------------------------
            num_agent, batch_size = agent_query_i.shape[:2]
            #agent_query_i = (agent_query_i.unsqueeze(1).repeat(1, self.fut_mode, 1, 1).flatten(0, 1)) # Repeat A_i so that it is [M,B,D]
            #agent_mask_i  = agent_mask_i.repeat(self.fut_mode,1)
            #dt = self.get_time_delay(img_metas)[0][0]
            #agent_query_i = agent_query_i.permute(1,0,2) + dt*ca_motion_query_i # [B,M,D]

            
            # A_i = A_i + M_i * dt
            # reshape A_i -> (B,A,fut_mode,D)
            # hungarian algo on cos similatiry to filter out seen objects from infra: [Av,Ai,fut_mode]
            # pick random fut-mode on remaining from infra #[A_new, B, D]

            
            # Agent Query fusion
            infra_tmp_list = []
            dt_sync = None
            use_time_comp = (
                self.v2x_use_time_compensation
                and ((not self.training) or getattr(self, 'epoch', 0) >= self.v2x_time_sync_start_epoch)
            )
            if use_time_comp:
                dt_sync = self.get_time_delay(img_metas).to(device=agent_query_i.device)

            for i in range(len(infra_global_coords_list)):
                infra_reference = infra_global_coords_list[i].clone()

                if (use_time_comp and infra_global_vels_list is not None
                        and i < len(infra_global_vels_list)):
                    # Align infrastructure detections from infra timestamp to vehicle
                    # timestamp. get_time_delay returns infra_ts - vehicle_ts, so
                    # x_vehicle_time = x_infra_time - v_global * dt.
                    dt_value = dt_sync[0, 0].to(
                        device=infra_reference.device, dtype=infra_reference.dtype)
                    dt_value_clamped = dt_value.clamp(
                        -self.v2x_time_sync_max_dt, self.v2x_time_sync_max_dt)

                    # Score gate: only shift queries that the infra head detects
                    # with reasonable confidence. Shifting all 900 queries adds
                    # ~870 noisy background shifts; restricting to confident
                    # detections focuses the correction where it matters.
                    det_score_thresh = self.v2x_time_sync_score_thresh
                    infra_cls_i = infra_classes[i]   # [1, Q, num_classes]
                    det_scores = infra_cls_i[0].sigmoid().max(dim=-1)[0]  # [Q]
                    confident = (det_scores > det_score_thresh).unsqueeze(-1)  # [Q, 1]

                    _R_veh2global = torch.as_tensor(
                        T_veh[:2, :2],
                        device=infra_reference.device,
                        dtype=infra_reference.dtype,
                    )  # [2, 2]

                    _pred_vel_veh = infra_global_vels_list[i].to(
                        device=infra_reference.device, dtype=infra_reference.dtype)  # [Q, 2]
                    global_vel = _pred_vel_veh @ _R_veh2global.T  # [Q, 2] predicted velocity in global frame

                    if global_vel.numel() > 0 and global_vel.shape[-1] >= 2:
                        vel_xy = global_vel[..., :2]
                        if self.v2x_time_sync_detach:
                            vel_xy = vel_xy.detach()

                        finite = torch.isfinite(vel_xy).all(dim=-1, keepdim=True)
                        speed = torch.norm(vel_xy, dim=-1, keepdim=True)
                        max_speed = max(
                            self.v2x_time_sync_max_speed,
                            self.v2x_time_sync_min_speed + 1e-3)
                        speed_scale = (max_speed / speed.clamp_min(1e-6)).clamp(max=1.0)
                        vel_xy = vel_xy * speed_scale

                        # Gate: confident AND finite AND meaningfully moving.
                        # Queries below v2x_time_sync_min_speed are near-static; shifting
                        # them adds noise rather than correcting temporal misalignment.
                        # Use continuous confidence weighting so partially confident
                        # detections receive proportionally attenuated shifts instead of
                        # a hard binary gate — this smoothly interpolates between "no
                        # compensation" (low conf) and "full compensation" (high conf).
                        is_moving = speed >= self.v2x_time_sync_min_speed   # [Q, 1]
                        valid_vel = finite & confident & is_moving
                        conf_weight = det_scores.unsqueeze(-1).clamp(0.0, 1.0)  # [Q, 1]
                        shift = torch.where(
                            valid_vel,
                            vel_xy * dt_value_clamped * conf_weight,
                            torch.zeros_like(vel_xy))
                        if i == 0:
                            n_conf = confident.squeeze(-1).sum().item()
                            n_moving_log = is_moving.squeeze(-1).sum().item()
                            n_valid = valid_vel.squeeze(-1).sum().item()
                            conf_mask_1d = (confident & finite).squeeze(-1)  # [Q]

                            # --- speed/shift stats for confident queries only ---
                            if conf_mask_1d.any():
                                mean_spd = speed[conf_mask_1d.unsqueeze(-1)].mean().item()
                                conf_shift = shift[conf_mask_1d]           # [n_conf, 2]
                                mean_shift_conf = conf_shift.norm(dim=-1).mean().item()
                                # Velocity direction in global frame (degrees from +X axis)
                                conf_vel = vel_xy[conf_mask_1d]            # [n_conf, 2]
                                mean_vel_angle = float(
                                    torch.atan2(conf_vel[:, 1], conf_vel[:, 0])
                                    .mean().item() * 180.0 / 3.14159)
                            else:
                                mean_spd = mean_shift_conf = mean_vel_angle = 0.0

                            # --- compare T_inf vs T_veh rotation on raw predictions ---
                            _pred_raw = infra_global_vels_list[i].to(
                                device=infra_reference.device,
                                dtype=infra_reference.dtype)
                            if _pred_raw.numel() > 0 and conf_mask_1d.any():
                                _R_inf_diag = torch.as_tensor(
                                    self._diag_T_inf_cache[:2, :2],
                                    device=_pred_raw.device,
                                    dtype=_pred_raw.dtype) if hasattr(self, '_diag_T_inf_cache') \
                                    else _R_veh2global  # fallback
                                _vel_via_Tinf = (_pred_raw[conf_mask_1d] @ _R_inf_diag.T)
                                _vel_via_Tveh = (_pred_raw[conf_mask_1d] @ _R_veh2global.T)
                                _angle_Tinf = float(
                                    torch.atan2(_vel_via_Tinf[:, 1], _vel_via_Tinf[:, 0])
                                    .mean().item() * 180.0 / 3.14159)
                                _angle_Tveh = float(
                                    torch.atan2(_vel_via_Tveh[:, 1], _vel_via_Tveh[:, 0])
                                    .mean().item() * 180.0 / 3.14159)
                                _angle_diff = abs(_angle_Tinf - _angle_Tveh)
                                _angle_diff = min(_angle_diff, 360.0 - _angle_diff)
                            else:
                                _angle_Tinf = _angle_Tveh = _angle_diff = 0.0

                            mean_shift_all = shift.abs().mean().item()
                        infra_reference = torch.cat(
                            [infra_reference[:, :2] - shift, infra_reference[:, 2:3]],
                            dim=-1)
                    else:
                        pass
                elif i == 0 and use_time_comp:
                    pass

                T_veh_t = torch.as_tensor(
                    T_veh, device=infra_reference.device, dtype=infra_reference.dtype)
                T_global_to_local = torch.linalg.inv(T_veh_t)
                ones = infra_reference.new_ones((infra_reference.size(0), 1))
                infra_global_h = torch.cat([infra_reference, ones], dim=1)
                infra_local_h = infra_global_h @ T_global_to_local.transpose(-1, -2)
                infra_reference = infra_local_h[:, :3].unsqueeze(0).to(infra_global_h.device)

                tmp_i_raw = infra_tmps[i]
                infra_x = (infra_reference[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
                infra_y = (infra_reference[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
                infra_z = (infra_reference[..., 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
                infra_xy = torch.cat([infra_x, infra_y], dim=-1).clamp(1e-5, 1 - 1e-5)
                infra_z = infra_z.clamp(1e-5, 1 - 1e-5)
                infra_class = infra_classes[i]

                # Rotate heading and velocity from infra-local → vehicle frame so that
                # the infra loss is supervised against matching GT (vehicle-frame).
                if (infra_global_vels_list is not None
                        and i < len(infra_global_vels_list)
                        and _R_g2v_np is not None):
                    _dev = infra_reference.device
                    _dt = infra_reference.dtype
                    # Velocity: infra_global_vels_list[i] is already global-frame;
                    # rotate to vehicle frame with R_global→vehicle.
                    _gv = infra_global_vels_list[i].to(device=_dev, dtype=_dt)  # [Q,2]
                    _R_g2v = torch.as_tensor(_R_g2v_np, device=_dev, dtype=_dt)  # [2,2]
                    _vel_veh = _gv @ _R_g2v.T  # [Q, 2]
                    # Heading: add yaw offset (infra frame → vehicle frame).
                    _sin_i = tmp_i_raw[..., 6:7]   # [1, Q, 1]
                    _cos_i = tmp_i_raw[..., 7:8]
                    _yaw_v = torch.atan2(_sin_i, _cos_i) + _yaw_offset_inf_to_veh
                    tmp_i = torch.cat([
                        inverse_sigmoid(infra_xy),
                        tmp_i_raw[..., 2:4],            # w, l (physical, frame-independent)
                        inverse_sigmoid(infra_z),
                        tmp_i_raw[..., 5:6],            # h
                        torch.sin(_yaw_v), torch.cos(_yaw_v),
                        _vel_veh.unsqueeze(0),          # [1, Q, 2]
                    ], dim=-1)
                else:
                    tmp_i = torch.cat([
                        inverse_sigmoid(infra_xy),
                        tmp_i_raw[..., 2:4],
                        inverse_sigmoid(infra_z),
                        tmp_i_raw[..., 5:]], dim=-1)
                infra_tmp_list.append(tmp_i)
            infra_tmp = infra_tmp_list[-1]

            # ---- Diagnostic: check infra hi-conf positions in vehicle-local frame ----
            # This reveals whether the global→vehicle transform is placing infra
            # detections anywhere near vehicle detections before matching begins.
            _inf_cls_last = infra_classes[-1].sigmoid().squeeze(0)   # [Q, C]
            _inf_sc_last, _ = _inf_cls_last.max(dim=-1)              # [Q]
            _inf_hi_idx = (_inf_sc_last >= self.v2x_infra_conf_thresh).nonzero(as_tuple=True)[0]
            _veh_cls_last = veh_class.sigmoid().squeeze(0)
            _veh_sc_last, _ = _veh_cls_last.max(dim=-1)
            _veh_hi_idx = (_veh_sc_last >= self.v2x_infra_conf_thresh).nonzero(as_tuple=True)[0]
            if _inf_hi_idx.numel() > 0 and _veh_hi_idx.numel() > 0:
                # infra_tmp encodes vehicle-local positions after global→vehicle transform
                _itmp = infra_tmp.squeeze(0)      # [Q, code]
                _vtmp = veh_tmp[..., :].squeeze(0) if veh_tmp.dim() == 3 else veh_tmp
                # decode metric XY for infra hi-conf
                _ix = _itmp[_inf_hi_idx, 0].sigmoid() * (self.pc_range[3]-self.pc_range[0]) + self.pc_range[0]
                _iy = _itmp[_inf_hi_idx, 1].sigmoid() * (self.pc_range[4]-self.pc_range[1]) + self.pc_range[1]
                _vx = _vtmp[_veh_hi_idx, 0].sigmoid() * (self.pc_range[3]-self.pc_range[0]) + self.pc_range[0]
                _vy = _vtmp[_veh_hi_idx, 1].sigmoid() * (self.pc_range[4]-self.pc_range[1]) + self.pc_range[1]
                _inf_pos = torch.stack([_ix, _iy], dim=-1).detach()     # [n_inf_hi, 2]
                _veh_pos = torch.stack([_vx, _vy], dim=-1).detach()     # [n_veh_hi, 2]
                # nearest vehicle detection for each infra hi-conf query
                _cross = (_inf_pos[:, None, :] - _veh_pos[None, :, :]).norm(dim=-1)  # [ni, nv]
                _nn_dists, _ = _cross.min(dim=1)   # [ni]
                _nn_top5 = _nn_dists.topk(min(5, _nn_dists.numel()), largest=False).values
                _pre_fusion_min_nn = float(_nn_top5[0])   # diagnostic only (may be off due to double-sigmoid on veh_tmp)
            else:
                _pre_fusion_min_nn = float('inf')

            # FOV mask: infra_reference is [1, Q, 3] in vehicle-local frame after the
            # decoder loop. Queries outside ±51.2m get clamped to the boundary, creating
            # phantom detections and spurious near-boundary matches. Exclude them from
            # both the matching cost and the unmatched detection pool.
            _inf_xv = infra_reference[0, :, 0]   # [Q] vehicle-local x
            _inf_yv = infra_reference[0, :, 1]   # [Q] vehicle-local y
            _infra_in_fov = (
                (_inf_xv >= self.pc_range[0]) & (_inf_xv <= self.pc_range[3]) &
                (_inf_yv >= self.pc_range[1]) & (_inf_yv <= self.pc_range[4])
            )  # [Q] bool

            # Agent Query fusion
            veh_tmp = torch.cat([
                inverse_sigmoid(veh_tmp[..., 0:2].clamp(1e-5, 1 - 1e-5)),
                veh_tmp[..., 2:4],
                inverse_sigmoid(veh_tmp[..., 4:5].clamp(1e-5, 1 - 1e-5)),
                veh_tmp[..., 5:]], dim=-1)

            
            #v = agent_query_v.squeeze(0)        # [900, 256]
            #i = agent_query_i.squeeze(1)      # [900, 256]
            #v_norm = F.normalize(v, dim=-1)   # [900, 256]
            #i_norm = F.normalize(i, dim=-1)   # [900, 256]
            #sim = v_norm @ i_norm.T
            #cost = -sim
            #row_idx, col_idx = linear_sum_assignment(cost.cpu().detach().numpy())
            #row_idx = torch.as_tensor(row_idx, device=sim.device, dtype=torch.long)
            #col_idx = torch.as_tensor(col_idx, device=sim.device, dtype=torch.long)
            #sim_thresh = 0.2
            #matched_sim = sim[row_idx, col_idx]
            #valid_sim = matched_sim > sim_thresh
            
            # Use the configured matching threshold (previously this was hard-coded to 2.0
            # and the config value v2x_match_dist_thresh was silently dropped into **kwargs).
            dist_thresh = self.v2x_match_dist_thresh

            veh_xy = torch.cat([veh_tmp[...,:2], veh_tmp[...,4:5]], dim=2).squeeze(0)    # logits
            inf_xy = torch.cat([infra_tmp[...,:2], infra_tmp[...,4:5]], dim=2).squeeze(0)# logits
            veh_xy = veh_xy[:, :2].sigmoid().clone()
            inf_xy = inf_xy[:, :2].sigmoid().clone()

            veh_xy[:,0] = veh_xy[:,0] * (self.pc_range[3]-self.pc_range[0]) + self.pc_range[0]
            veh_xy[:,1] = veh_xy[:,1] * (self.pc_range[4]-self.pc_range[1]) + self.pc_range[1]
            inf_xy[:,0] = inf_xy[:,0] * (self.pc_range[3]-self.pc_range[0]) + self.pc_range[0]
            inf_xy[:,1] = inf_xy[:,1] * (self.pc_range[4]-self.pc_range[1]) + self.pc_range[1]

            diff = veh_xy[:, None, :] - inf_xy[None, :, :]
            dist2 = (diff ** 2).sum(dim=-1)                               # [Qv,Qi]

            infra_class = infra_classes[-1]
            veh_prob = veh_class.sigmoid().squeeze(0)
            inf_prob = infra_class.sigmoid().squeeze(0)
            veh_score, veh_label = veh_prob.max(dim=-1)
            inf_score, inf_label = inf_prob.max(dim=-1)

            same_cls = veh_label[:, None] == inf_label[None, :]

            cost = dist2.clone()
            cost = cost + self.v2x_match_cls_cost * (1.0 - (same_cls.float()))
            cost = cost.masked_fill(~same_cls, 1e6)
            cost = cost.masked_fill(dist2 > dist_thresh**2, 1e6)

            # Confidence pre-filter: only allow matching between hi-conf queries.
            veh_match_mask = veh_score >= self.v2x_infra_conf_thresh   # [Qv]
            inf_match_mask = inf_score >= self.v2x_infra_conf_thresh   # [Qi]
            low_conf_pair = (~veh_match_mask[:, None]) | (~inf_match_mask[None, :])
            cost = cost.masked_fill(low_conf_pair, 1e6)

            # FOV filter: exclude out-of-FOV infra queries from matching.
            # These clamp to ±51.2m in inf_xy and could spuriously match vehicle queries
            # near the BEV boundary, biasing fused positions toward the edge.
            cost = cost.masked_fill(~_infra_in_fov[None, :], 1e6)

            # Hungarian (SciPy wants CPU numpy)
            row, col = linear_sum_assignment(cost.detach().cpu().numpy())
            row_idx = torch.as_tensor(row, device=veh_xy.device, dtype=torch.long)
            col_idx = torch.as_tensor(col, device=veh_xy.device, dtype=torch.long)
            valid = cost[row, col] < 1e6 * 0.5

            veh_idx = torch.tensor(row_idx)[valid]
            infra_idx = torch.tensor(col_idx)[valid]

            matches = list(zip(veh_idx.tolist(), infra_idx.tolist()))

            # ---- Diagnostic: match quality breakdown ----
            if len(matches) > 0:
                matched_veh_scores = veh_score[veh_idx]
                matched_inf_scores = inf_score[infra_idx]
                n_both_hi  = int(((matched_veh_scores >= self.v2x_infra_conf_thresh) &
                                  (matched_inf_scores >= self.v2x_infra_conf_thresh)).sum())
                n_veh_only = int(((matched_veh_scores >= self.v2x_infra_conf_thresh) &
                                  (matched_inf_scores <  self.v2x_infra_conf_thresh)).sum())
                n_inf_only = int(((matched_veh_scores <  self.v2x_infra_conf_thresh) &
                                  (matched_inf_scores >= self.v2x_infra_conf_thresh)).sum())
                n_neither  = int(((matched_veh_scores <  self.v2x_infra_conf_thresh) &
                                  (matched_inf_scores <  self.v2x_infra_conf_thresh)).sum())
                matched_dists = torch.norm(veh_xy[veh_idx] - inf_xy[infra_idx], dim=-1)
                mean_dist = matched_dists.mean().item()
                max_dist  = matched_dists.max().item()
            else:
                n_both_hi = n_veh_only = n_inf_only = n_neither = 0
                mean_dist = max_dist = 0.0

            # ---- Diagnostic: spatial alignment sanity (top-5 nearest hi-conf pairs) ----
            hi_v = veh_match_mask.nonzero(as_tuple=True)[0]   # vehicle hi-conf indices
            hi_i = inf_match_mask.nonzero(as_tuple=True)[0]   # infra hi-conf indices
            if hi_v.numel() > 0 and hi_i.numel() > 0:
                sub_diff = veh_xy[hi_v][:, None, :] - inf_xy[hi_i][None, :, :]  # [nv, ni, 2]
                sub_dist = sub_diff.norm(dim=-1)                                 # [nv, ni]
                min_dists, _ = sub_dist.min(dim=1)   # nearest infra for each vehicle hi-conf
                top5 = min_dists.topk(min(5, min_dists.numel()), largest=False).values
                nn_summary = '  '.join(f'{d:.2f}m' for d in top5.tolist())
            else:
                nn_summary = 'no hi-conf pairs'


            
            
            #fuse infra-vehicle queries
            #agent_query_fusion = self.vi_agent_fuser(
                #query=agent_query_v[:, veh_idx, :].permute(1, 0, 2),
                #key=agent_query_i[infra_idx, :, :].permute(1,0,2), # TODO: [A_i,B,D] -> [M_i,B,D] M = A * fut_mode
                #value=agent_query_i[infra_idx, :, :].permute(1,0,2),
                #query_pos= None, #veh_agent_pos_embed.permute(1, 0, 2),
                #key_pos=None, #infra_agent_pos_embed.permute(1, 0, 2),
                #key_padding_mask=agent_mask_i[infra_idx, :].permute(1,0).bool()
            #) #[A_v,B,D]
            
            
            infra_unmatched_idx = []
            _, Nv, D = agent_query_v.shape
            Ni = agent_query_i.shape[0]

            # infra unmatched → remove infra_idx
            probs = torch.softmax(infra_classes[-1], dim=-1)
            confidence, labels = probs.max(dim=-1)

            # Confidence-weighted blend: matched pairs merged by detection score.
            if veh_idx.numel() > 0:
                _wv = veh_score[veh_idx].clamp(1e-6, 1.0)   # [M]
                _wi = inf_score[infra_idx].clamp(1e-6, 1.0)  # [M]
                _w_sum = _wv + _wi
                _wv_norm = (_wv / _w_sum).view(1, -1, 1)     # [1, M, 1]
                _wi_norm = (_wi / _w_sum).view(1, -1, 1)     # [1, M, 1]
            else:
                _wv_norm = _wi_norm = torch.tensor(0.5, device=veh_tmp.device)
            # Plain addition — cross_agent_fusion was trained at scale 1.0;
            # confidence-weighted addition should be trained in, not applied cold.
            agent_query_fusion = agent_query_v[:, veh_idx, :].permute(1,0,2) + self.cross_agent_fusion(agent_query_i[infra_idx, :, :])
            agent_query_fusion = agent_query_fusion.permute(1,0,2)

            fusion_tmp = _wv_norm * veh_tmp[:, veh_idx, :] + _wi_norm * infra_tmp[:, infra_idx, :]
            fusion_class = veh_class[:, veh_idx, :]
            fusion_pos_emb = _wv_norm * agent_pos_emb[:, veh_idx, :] + _wi_norm * agent_pos_emb_i[:, infra_idx, :]

            for i in range(Ni):
                if i not in infra_idx and confidence[:, i] > 0.85 and _infra_in_fov[i].item():
                    infra_unmatched_idx.append(i)
            n_infra_unmatched = len(infra_unmatched_idx)
            agent_query_i_unmatched = agent_query_i[infra_unmatched_idx,:,:].permute(1,0,2)
            infra_unmatched_tmp = infra_tmp[:, infra_unmatched_idx, :]
            infra_unmatched_class = infra_class[:, infra_unmatched_idx, :]
                    
            cur_len = agent_query_fusion.shape[1] + agent_query_i_unmatched.shape[1]
                
            veh_unmatched_idx = torch.tensor(list(set(range(Nv)) - set(veh_idx.tolist())))
            agent_query_v_unmatched = agent_query_v[:,veh_unmatched_idx,:]
            veh_unmatched_tmp = veh_tmp[:, veh_unmatched_idx, :]
            veh_unmatched_class = veh_class[:, veh_unmatched_idx, :]
            agent_query = torch.cat([agent_query_fusion, agent_query_v_unmatched, agent_query_i_unmatched], dim=1)
            
            agent_pos = torch.cat([fusion_pos_emb, agent_pos_emb[:,veh_unmatched_idx,:], agent_pos_emb_i[:,infra_unmatched_idx,:]], dim=1)
            tmp = torch.cat([fusion_tmp, veh_unmatched_tmp, infra_unmatched_tmp], dim=1)
           
            
            outputs_class = torch.cat([fusion_class, veh_unmatched_class, infra_unmatched_class], dim=1)
                

            # Anchor boxes fusion
            
            
            if self.use_infra_map and self.vi_map_fuser is not None:
                map_query = self.vi_map_fuser(
                    query=map_query.permute(1, 0, 2), #[P_v,B,D]
                    key=map_query_i.permute(1, 0, 2), #[P_i, B, D]
                    value=map_query_i.permute(1, 0, 2),
                    query_pos= None,
                    key_pos= None,
                    key_padding_mask=map_mask_i
                ).permute(1,0,2) #[B,P_v,D]
            # else: keep map_query as pure vehicle map (no infra influence)


        else:
            n_infra_unmatched = 0
            agent_query = agent_query.permute(1,0,2)
            reference = inter_references[-1]
            #reference = inverse_sigmoid(reference)

        
        # Do classification and regression of map/agent classes ans coordinates from fused queries
        outputs_classes = []
        outputs_coords = []
        outputs_coords_bev = []
        outputs_trajs = []
        outputs_trajs_classes = []

        map_outputs_classes = []
        map_outputs_coords = []
        map_outputs_pts_coords = []
        map_outputs_coords_bev = []

        reference = torch.cat([tmp[...,:2], tmp[...,4:5]], dim=2)
        bs = bev_embed.shape[1]
        
        query_pos = agent_pos
        query = agent_query
        reference_points = reference
        reference_points = reference_points.sigmoid().clone()
        init_reference = reference_points
        query = query.permute(1,0,2)
        query_pos = query_pos.permute(1,0,2)

        #path = os.path.join('/scratch/jmeng18/bev_emb/', f"bev_embed_{img_metas[0]['pts_filename'][-10:-4]}.pt")
        #bev_embed_true = torch.load(path, map_location="cuda:0")

        #path = os.path.join('/scratch/jmeng18/bev_emb/', f"bev_pose_{img_metas[0]['pts_filename'][-10:-4]}.pt")
        #bev_pos_true = torch.load(path, map_location="cuda:0")

        #path = os.path.join('/scratch/jmeng18/bev_emb/', f"query_{img_metas[0]['pts_filename'][-10:-4]}.pt")
        #query_true = torch.load(path, map_location="cuda:0")
        
        
        bev_embed, bev_pos = self._get_coop_bev_embed(bev_embed, bev_pos, query, query_pos, reference_points, -len(infra_unmatched_idx))
        
        
        hs, inter_references = self.agent_fusion_decoder(
            query=query,
            key = None,
            value = bev_embed,
            query_pos = query_pos,
            reference_points = reference_points,
            reg_branches=self.reg_branches_fuse,
            cls_branches=self.cls_branches_fuse,
            spatial_shapes=torch.tensor([[self.bev_h, self.bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            img_metas=img_metas)
        
        hs = hs.permute(0,2,1,3)
        
        
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl-1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches_fuse[lvl](hs[lvl])
            tmp = self.reg_branches_fuse[lvl](hs[lvl])
            #tmp = torch.zeros_like(tmp)
            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            
            tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid().clone()
            outputs_coords_bev.append(tmp[..., 0:2].clone().detach())
            tmp[..., 4:5] = tmp[..., 4:5] + reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid().clone()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -self.pc_range[2]) + self.pc_range[2])

            # TODO: check if using sigmoid
            
            
            
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        #tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
        #tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
        #outputs_coords_bev.append(tmp[..., 0:2].clone().detach())
        #tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                             #self.pc_range[0]) + self.pc_range[0])
        #tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                             #self.pc_range[1]) + self.pc_range[1])
        #tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                             #self.pc_range[2]) + self.pc_range[2])

        
        #outputs_coord = tmp[:,900:,:]
        #outputs_classes.append(outputs_class[:,900:,:])
        #outputs_coords.append(outputs_coord)
        #agent_query = agent_query_v
        #agent_pos = agent_pos_emb
        #outputs_coord = tmp
        #outputs_classes.append(outputs_class)
        #outputs_coords.append(outputs_coord)
        
        #import pdb; pdb.set_trace()
        #outputs_class = self.cls_branches_fuse[-1](agent_query.permute(1,0,2))#self.cls_branches(agent_query)
        #tmp = self.reg_branches_fuse[-1](agent_query.permute(1,0,2))

        
        #reference = inverse_sigmoid(reference)
        #tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2] # denormalized one, inverse_sigmoid
        #tmp[..., 4:5] = tmp[..., 4:5] + reference[..., 2:3]
        #tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
        #tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
        #outputs_coords_bev.append(tmp[..., 0:2].clone().detach())
        #import pdb; pdb.set_trace()
        #tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        #tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        #tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])
        #import pdb; pdb.set_trace()
        #xy = tmp[...,:2].squeeze(0).cpu().detach().numpy()
        #plt.figure(figsize=(6,6))
        #plt.scatter(xy[:, 0], xy[:, 1], s=5)   # s = point size
        #plt.xlabel("X")
        #plt.ylabel("Y")
        #plt.grid(True)
        #plt.show()
        
        #print(tmp[:, agent_query_fusion.shape[1]:(agent_query_fusion.shape[1] + agent_query_i_unmatched.shape[1]), 0])
        #print(tmp[:, agent_query_fusion.shape[1]:(agent_query_fusion.shape[1] + agent_query_i_unmatched.shape[1]), 1])
        #import pdb; pdb.set_trace()
        

        reference = map_inter_references[-1]#map_init_reference
        reference = inverse_sigmoid(reference)
        
        # Use per-point hidden states from the last map decoder layer directly.
        # map_hs[-1] shape: [num_vec*num_pts, B, D] — each of the 2000 slots
        # has a distinct feature vector encoding its specific point context.
        # Repeating the per-vector aggregated map_query would make all 10 points
        # for each vector identical, collapsing the prediction to a single point.
        B = map_hs[-1].shape[1]
        map_query_tmp = map_hs[-1].permute(1, 0, 2)  # [B, num_vec*num_pts, D]
 
        map_outputs_class = self.map_cls_branches[-1]((
                map_query_tmp.view(B, self.map_num_vec, self.map_num_pts_per_vec, -1).mean(2)
            ))
        tmp = self.map_reg_branches[-1](map_query_tmp)
        # reference = map_inter_references[-1]#map_init_reference
        tmp[..., 0:2] = tmp[..., 0:2]+reference[..., 0:2]
        tmp[..., 0:2] = tmp[..., 0:2].sigmoid() # cx,cy,w,h
        map_outputs_coord, map_outputs_pts_coord = self.map_transform_box(tmp)
        map_outputs_coords_bev.append(map_outputs_pts_coord.clone().detach())
        map_outputs_classes.append(map_outputs_class)
        map_outputs_coords.append(map_outputs_coord)
        map_outputs_pts_coords.append(map_outputs_pts_coord)

        

        # ----------------------------- Map masking -------------------------------------
        motion_coords = outputs_coords_bev[-1]  # [B, A, 2]
        motion_coords = motion_coords.unsqueeze(2).repeat(1, 1, self.fut_mode, 1).flatten(1, 2)
        map_conf = map_outputs_classes[-1]
        map_pos_raw = map_outputs_coords_bev[-1]

        # Per-agent map masking:
        #   - removes road-edge class (index 2) and low-confidence map elements
        #   - masks out map elements within dis_thresh of each agent (the road
        #     segment the agent is currently on) so motion cross-attention focuses
        #     on reachable road ahead rather than the agent's current position
        # map_query_masked [B*M, P, D], map_mask_masked [B*M, P]: per agent-mode
        # map_query_ego    [B,   P, D], map_mask_ego    [B,   P]: batch-level for ego
        map_query_masked, map_pos_masked, map_mask_masked, map_query_ego, map_pos_ego, map_mask_ego = self.select_and_mask_map(
            motion_pos=motion_coords,
            map_query=map_query.detach(),
            map_score=map_conf,
            map_pos_pts=map_pos_raw,
            map_thresh=self.map_thresh,
            dis_thresh=self.dis_thresh,
            road_edge_idx=2,
            pe_normalization=False,
            use_fix_pad=True)

        # VI motion queries
        # motion SA
        
        motion_query = self.vi_motion(
            query=agent_query.permute(1, 0, 2),
            key=agent_query.permute(1, 0, 2),
            value=agent_query.permute(1, 0, 2),
            query_pos=None,#veh_agent_pos_embed,
            key_pos=None,#veh_agent_pos_embed,
            key_padding_mask=None #agent_mask.bool()
        ).permute(1,0,2) #[A, B, D]

        mode_query = self.motion_mode_query.weight  # [fut_mode, D]
        # [M, B, D], M=A*fut_mode
        motion_query = (motion_query[:, None, :, :] + mode_query[None, :, None, :]).flatten(0, 1)
        if self.use_pe:
            motion_coords = outputs_coords_bev[-1]  # [B, A, 2]
            motion_pos = self.pos_mlp_sa(motion_coords)  # [B, A, D]
            motion_pos = motion_pos.unsqueeze(2).repeat(1, 1, self.fut_mode, 1).flatten(1, 2)
            motion_pos = motion_pos.permute(1, 0, 2)  # [M, B, D]
        else:
            motion_pos = None

        if self.motion_det_score is not None:
            motion_score = outputs_classes[-1]
            max_motion_score = motion_score.max(dim=-1)[0]
            invalid_motion_idx = max_motion_score < self.motion_det_score  # [B, A]
            invalid_motion_idx = invalid_motion_idx.unsqueeze(2).repeat(1, 1, self.fut_mode).flatten(1, 2)
        else:
            invalid_motion_idx = None


        ca_motion_query = motion_query.permute(1, 0, 2).flatten(0, 1).unsqueeze(0) #[B,M,D]
        # motion CA
        
        ca_motion_query = self.vi_map_motion(
            query=ca_motion_query,
            key=map_query_masked.permute(1, 0, 2), #[P,B*M,D]
            value=map_query_masked.permute(1, 0, 2),
            query_pos=motion_pos.permute(1,0,2),#veh_agent_pos_embed.permute(1, 0, 2),
            key_pos=None,#veh_map_pos_embed_masked.permute(1,0,2),
            key_padding_mask=map_mask_masked
        )

        batch_size, num_agent = outputs_coords_bev[-1].shape[:2]

        
        motion_query = motion_query.permute(1, 0, 2)
        #motion_query = motion_query.squeeze(0).permute(1, 0, 2).unflatten(
                #dim=1, sizes=(num_agent, self.fut_mode))
        ca_motion_query = ca_motion_query.permute(1, 0, 2)
        #ca_motion_query = ca_motion_query.squeeze(0).unflatten(
                #dim=0, sizes=(batch_size, num_agent, self.fut_mode)
            #)
        ca_motion_query = ca_motion_query.squeeze(1)
        ca_motion_query = ca_motion_query.view(num_agent, self.fut_mode, -1)
        
        motion_hs = torch.cat([motion_query, ca_motion_query], dim=-1) # [B, A, fut_mode, 2D]
        

        # make agent query from motion_hs like the original VAD_head

        outputs_traj = self.traj_branches[0](motion_hs)
        outputs_trajs.append(outputs_traj)
        outputs_traj_class = self.traj_cls_branches[0](motion_hs)
        outputs_trajs_classes.append(outputs_traj_class.squeeze(-1))
             
        map_outputs_classes = torch.stack(map_outputs_classes)
        map_outputs_coords = torch.stack(map_outputs_coords)
        map_outputs_pts_coords = torch.stack(map_outputs_pts_coords)
        
        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_trajs = torch.stack(outputs_trajs)
        outputs_trajs_classes = torch.stack(outputs_trajs_classes)

        (batch, num_agent) = motion_hs.shape[:2]
        agent_conf = outputs_classes[-1]
        agent_query = motion_hs.reshape(batch, num_agent, -1)
        A, M, D = agent_query.shape
        agent_query_flat = agent_query.reshape(A, M * D)
        agent_query = self.agent_fus_mlp(agent_query_flat) # [B, A, fut_mode, 2*D] -> [B, A, D]
        agent_pos = outputs_coords_bev[-1]
        
        agent_query, agent_pos, agent_mask = self.select_and_pad_query(
            agent_query.unsqueeze(0), agent_pos, agent_conf,
            score_thresh=self.query_thresh, use_fix_pad=self.query_use_fix_pad
        )
        # ego <-> agent interaction
        # ego_query: [1 ,1, 256]  agent_query: [300, 1, 256]   map_query: [1, 100, 256]
        ego_agent_query = self.ego_agent_decoder(
            query=ego_query.permute(1, 0, 2),
            key=agent_query.permute(1, 0, 2),
            value=agent_query.permute(1, 0, 2),
            query_pos=ego_pos_emb.permute(1, 0, 2),
            key_pos=None,#veh_agent_pos_embed,
            key_padding_mask=agent_mask.bool() )

        # ego <-> map interaction
        #import pdb; pdb.set_trace()
        ego_map_query = self.ego_map_decoder(
            query=ego_agent_query,
            key=map_query_ego.permute(1,0,2),
            value=map_query_ego.permute(1,0,2),
            query_pos=ego_pos_emb,
            key_pos=None,#veh_map_pos_embed_masked.permute(1, 0, 2),
            key_padding_mask=map_mask_ego )

        # Concat the Q', Q'', current status of ego vehicle
        if self.ego_his_encoder is not None and self.ego_lcf_feat_idx is not None:
            ego_feats = torch.cat(
                [ego_his_feats,
                 ego_map_query.permute(1, 0, 2),
                 ego_lcf_feat.squeeze(1)[..., self.ego_lcf_feat_idx]],
                dim=-1
            )  # [B, 1, 2D+2]
        elif self.ego_his_encoder is not None and self.ego_lcf_feat_idx is None:
            ego_feats = torch.cat(
                [ego_his_feats,
                 ego_map_query.permute(1, 0, 2)],
                dim=-1
            )  # [B, 1, 2D]
        elif self.ego_his_encoder is None and self.ego_lcf_feat_idx is not None:                
            ego_feats = torch.cat(
                [ego_agent_query.permute(1, 0, 2),
                 ego_map_query.permute(1, 0, 2),
                 ego_lcf_feat.squeeze(1)[..., self.ego_lcf_feat_idx]],
                dim=-1
            )  # [B, 1, 2D+2]
        elif self.ego_his_encoder is None and self.ego_lcf_feat_idx is None:                
            ego_feats = torch.cat(
                [ego_agent_query.permute(1, 0, 2),
                 ego_map_query.permute(1, 0, 2)],
                dim=-1
            )  # [B, 1, 2D]  

        # Ego prediction
        outputs_ego_trajs = self.ego_fut_decoder(ego_feats)
        
        outputs_ego_trajs = outputs_ego_trajs.reshape(outputs_ego_trajs.shape[0], 
                                                      self.ego_fut_mode, self.fut_ts, 2)


        infra_pred_coords_list = []
        for i in range(len(infra_tmp_list)):
            infra_pred_coords = infra_tmp_list[i]
            infra_pred_coords[..., 0:2] = infra_pred_coords[...,0:2].sigmoid()
            infra_pred_coords[..., 4:5] = infra_pred_coords[..., 4:5].sigmoid()

            infra_pred_coords[..., 0:1] = (infra_pred_coords[..., 0:1] * (self.pc_range[3] -self.pc_range[0]) + self.pc_range[0])
            infra_pred_coords[..., 1:2] = (infra_pred_coords[..., 1:2] * (self.pc_range[4] -self.pc_range[1]) + self.pc_range[1])
            infra_pred_coords[..., 4:5] = (infra_pred_coords[..., 4:5] * (self.pc_range[5] -self.pc_range[2]) + self.pc_range[2])
            infra_pred_coords_list.append(infra_pred_coords)

        infra_veh_pc_range = infra_global_pc_range
        if infra_global_pc_range is not None:  # convert infra global AABB into vehicle frame
            # infra_global_pc_range is an AABB [xmin, ymin, zmin, xmax, ymax, zmax],
            # not homogeneous points. Do not multiply the 6-vector by a 4x4 matrix.
            T_global_to_veh = np.linalg.inv(T_veh)
            infra_veh_pc_range = self.transform_pc_range_aabb(
                infra_global_pc_range, T_global_to_veh)
        outs = {
            'bev_embed': bev_embed.clone(),
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'all_traj_preds': outputs_trajs.repeat(outputs_coords.shape[0], 1, 1, 1, 1),
            'all_traj_cls_scores': outputs_trajs_classes.repeat(outputs_coords.shape[0], 1, 1, 1),
            'map_all_cls_scores': map_outputs_classes,
            'map_all_bbox_preds': map_outputs_coords,
            'map_all_pts_preds': map_outputs_pts_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            'map_enc_cls_scores': None,
            'map_enc_bbox_preds': None,
            'map_enc_pts_preds': None,
            'ego_fut_preds': outputs_ego_trajs,
            'infra_pred_coords':infra_pred_coords_list,
            'infra_pred_class': infra_classes,
            'infra_veh_pc_range': infra_veh_pc_range,
            'n_infra_unmatched': n_infra_unmatched,
        }
        return outs

    def map_transform_box(self, pts, y_first=False):
        """
        Converting the points set into bounding box.

        Args:
            pts: the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            The bbox [cx, cy, w, h] transformed from points.
        """
        pts_reshape = pts.view(pts.shape[0], self.map_num_vec,
                                self.map_num_pts_per_vec,2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.map_transform_method == 'minmax':

            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError
        return bbox, pts_reshape

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_attr_labels,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 10].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 9) in [x,y,z,w,l,h,yaw,vx,vy] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_fut_trajs = gt_attr_labels[:, :self.fut_ts*2]
        gt_fut_masks = gt_attr_labels[:, self.fut_ts*2:self.fut_ts*3]
        gt_bbox_c = gt_bboxes.shape[-1]
        num_gt_bbox, gt_traj_c = gt_fut_trajs.shape

        
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_bbox_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # trajs targets
        traj_targets = torch.zeros((num_bboxes, gt_traj_c), dtype=torch.float32, device=bbox_pred.device)
        traj_weights = torch.zeros_like(traj_targets)
        traj_targets[pos_inds] = gt_fut_trajs[sampling_result.pos_assigned_gt_inds]
        traj_weights[pos_inds] = 1.0

        # Filter out invalid fut trajs
        traj_masks = torch.zeros_like(traj_targets)  # [num_bboxes, fut_ts*2]
        gt_fut_masks = gt_fut_masks.unsqueeze(-1).repeat(1, 1, 2).view(num_gt_bbox, -1)  # [num_gt_bbox, fut_ts*2]
        traj_masks[pos_inds] = gt_fut_masks[sampling_result.pos_assigned_gt_inds]
        traj_weights = traj_weights * traj_masks

        # Extra future timestamp mask for controlling pred horizon
        fut_ts_mask = torch.zeros((num_bboxes, self.fut_ts, 2),
                                   dtype=torch.float32, device=bbox_pred.device)
        fut_ts_mask[:, :self.valid_fut_ts, :] = 1.0
        fut_ts_mask = fut_ts_mask.view(num_bboxes, -1)
        traj_weights = traj_weights * fut_ts_mask

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        return (
            labels, label_weights, bbox_targets, bbox_weights, traj_targets,
            traj_weights, traj_masks.view(-1, self.fut_ts, 2)[..., 0],
            pos_inds, neg_inds
        )

    def _map_get_target_single(self,
                           cls_score,
                           bbox_pred,
                           pts_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_shifts_pts,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]
        assign_result, order_index = self.map_assigner.assign(bbox_pred, cls_score, pts_pred,
                                             gt_bboxes, gt_labels, gt_shifts_pts,
                                             gt_bboxes_ignore)

        sampling_result = self.map_sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.map_num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)
        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        # pts targets
        if order_index is None:
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]
        pts_targets = pts_pred.new_zeros((pts_pred.size(0),
                        pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0
        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds,assigned_shift,:,:]
        return (labels, label_weights, bbox_targets, bbox_weights,
                pts_targets, pts_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_attr_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, traj_targets_list, traj_weights_list,
         gt_fut_masks_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_attr_labels_list, gt_bboxes_ignore_list
         )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
                traj_targets_list, traj_weights_list, gt_fut_masks_list, num_total_pos, num_total_neg)

    def map_get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    pts_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pts_targets_list, pts_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._map_get_target_single, cls_scores_list, bbox_preds_list,pts_preds_list,
            gt_labels_list, gt_bboxes_list, gt_shifts_pts_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, pts_targets_list, pts_weights_list,
                num_total_pos, num_total_neg)

    def loss_planning(self,
                      ego_fut_preds,
                      ego_fut_gt,
                      ego_fut_masks,
                      ego_fut_cmd,
                      lane_preds,
                      lane_score_preds,
                      agent_preds,
                      agent_fut_preds,
                      agent_score_preds,
                      agent_fut_cls_preds):
        """"Loss function for ego vehicle planning.
        Args:
            ego_fut_preds (Tensor): [B, ego_fut_mode, fut_ts, 2]
            ego_fut_gt (Tensor): [B, fut_ts, 2]
            ego_fut_masks (Tensor): [B, fut_ts]
            ego_fut_cmd (Tensor): [B, ego_fut_mode]
            lane_preds (Tensor): [B, num_vec, num_pts, 2]
            lane_score_preds (Tensor): [B, num_vec, 3]
            agent_preds (Tensor): [B, num_agent, 2]
            agent_fut_preds (Tensor): [B, num_agent, fut_mode, fut_ts, 2]
            agent_score_preds (Tensor): [B, num_agent, 10]
            agent_fut_cls_scores (Tensor): [B, num_agent, fut_mode]
        Returns:
            loss_plan_reg (Tensor): planning reg loss.
            loss_plan_bound (Tensor): planning map boundary constraint loss.
            loss_plan_col (Tensor): planning col constraint loss.
            loss_plan_dir (Tensor): planning directional constraint loss.
        """

        ego_fut_gt = ego_fut_gt.unsqueeze(1).repeat(1, self.ego_fut_mode, 1, 1)
        loss_plan_l1_weight = ego_fut_cmd[..., None, None] * ego_fut_masks[:, None, :, None]
        loss_plan_l1_weight = loss_plan_l1_weight.repeat(1, 1, 1, 2)

        loss_plan_l1 = self.loss_plan_reg(
            ego_fut_preds,
            ego_fut_gt,
            loss_plan_l1_weight
        )

        loss_plan_bound = self.loss_plan_bound(
            ego_fut_preds[ego_fut_cmd==1],
            lane_preds,
            lane_score_preds,
            weight=ego_fut_masks
        )

        loss_plan_col = self.loss_plan_col(
            ego_fut_preds[ego_fut_cmd==1],
            agent_preds,
            agent_fut_preds,
            agent_score_preds,
            agent_fut_cls_preds,
            weight=ego_fut_masks[:, :, None].repeat(1, 1, 2)
        )

        loss_plan_dir = self.loss_plan_dir(
            ego_fut_preds[ego_fut_cmd==1],
            lane_preds,
            lane_score_preds,
            weight=ego_fut_masks
        )

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_plan_l1 = torch.nan_to_num(loss_plan_l1)
            loss_plan_bound = torch.nan_to_num(loss_plan_bound)
            loss_plan_col = torch.nan_to_num(loss_plan_col)
            loss_plan_dir = torch.nan_to_num(loss_plan_dir)
        
        loss_plan_dict = dict()
        loss_plan_dict['loss_plan_reg'] = loss_plan_l1
        loss_plan_dict['loss_plan_bound'] = loss_plan_bound
        loss_plan_dict['loss_plan_col'] = loss_plan_col
        loss_plan_dict['loss_plan_dir'] = loss_plan_dir

        return loss_plan_dict
    
    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    traj_preds,
                    traj_cls_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_attr_labels_list,
                    gt_bboxes_ignore_list=None,
                    n_infra_unmatched=0):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        # Exclude infra unmatched queries from the fusion loss.  The Hungarian
        # matcher would otherwise label them as background (they don't overlap
        # vehicle GT), gradually suppressing their confidence over epochs.
        if n_infra_unmatched > 0:
            cls_scores     = cls_scores    [:, :-n_infra_unmatched, :]
            bbox_preds     = bbox_preds    [:, :-n_infra_unmatched, :]
            traj_preds     = traj_preds    [:, :-n_infra_unmatched, ...]
            traj_cls_preds = traj_cls_preds[:, :-n_infra_unmatched, :]

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_attr_labels_list, gt_bboxes_ignore_list)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         traj_targets_list, traj_weights_list, gt_fut_masks_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        traj_targets = torch.cat(traj_targets_list, 0)
        traj_weights = torch.cat(traj_weights_list, 0)
        gt_fut_masks = torch.cat(gt_fut_masks_list, 0)

        # NaN diagnostic — prints once per NaN occurrence, then stops
        if not getattr(self, '_nan_reported', False):
            if not cls_scores.isfinite().all():
                print(f'[NaN] cls_scores has NaN/Inf: shape={cls_scores.shape}', flush=True)
                self._nan_reported = True
            if not bbox_preds.isfinite().all():
                print(f'[NaN] bbox_preds has NaN/Inf: shape={bbox_preds.shape}', flush=True)
                self._nan_reported = True

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)

        # traj regression loss
        best_traj_preds = self.get_best_fut_preds(
            traj_preds.reshape(-1, self.fut_mode, self.fut_ts, 2),
            traj_targets.reshape(-1, self.fut_ts, 2), gt_fut_masks)

        neg_inds = (bbox_weights[:, 0] == 0)
        traj_labels = self.get_traj_cls_target(
            traj_preds.reshape(-1, self.fut_mode, self.fut_ts, 2),
            traj_targets.reshape(-1, self.fut_ts, 2),
            gt_fut_masks, neg_inds)

        loss_traj = self.loss_traj(
            best_traj_preds[isnotnan],
            traj_targets[isnotnan],
            traj_weights[isnotnan],
            avg_factor=num_total_pos)

        if self.use_traj_lr_warmup:
            loss_scale_factor = get_traj_warmup_loss_weight(self.epoch, self.tot_epoch)
            loss_traj = loss_scale_factor * loss_traj

        # traj classification loss
        traj_cls_scores = traj_cls_preds.reshape(-1, self.fut_mode)
        # construct weighted avg_factor to match with the official DETR repo
        traj_cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.traj_bg_cls_weight
        if self.sync_cls_avg_factor:
            traj_cls_avg_factor = reduce_mean(
                traj_cls_scores.new_tensor([traj_cls_avg_factor]))

        traj_cls_avg_factor = max(traj_cls_avg_factor, 1)
        loss_traj_cls = self.loss_traj_cls(
            traj_cls_scores, traj_labels, label_weights, avg_factor=traj_cls_avg_factor
        )

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_traj = torch.nan_to_num(loss_traj)
            loss_traj_cls = torch.nan_to_num(loss_traj_cls)

        return loss_cls, loss_bbox, loss_traj, loss_traj_cls

    def get_best_fut_preds(self,
             traj_preds,
             traj_targets,
             gt_fut_masks):
        """"Choose best preds among all modes.
        Args:
            traj_preds (Tensor): MultiModal traj preds with shape (num_box_preds, fut_mode, fut_ts, 2).
            traj_targets (Tensor): Ground truth traj for each pred box with shape (num_box_preds, fut_ts, 2).
            gt_fut_masks (Tensor): Ground truth traj mask with shape (num_box_preds, fut_ts).
            pred_box_centers (Tensor): Pred box centers with shape (num_box_preds, 2).
            gt_box_centers (Tensor): Ground truth box centers with shape (num_box_preds, 2).

        Returns:
            best_traj_preds (Tensor): best traj preds (min displacement error with gt)
                with shape (num_box_preds, fut_ts*2).
        """

        cum_traj_preds = traj_preds.cumsum(dim=-2)
        cum_traj_targets = traj_targets.cumsum(dim=-2)
        # Get min pred mode indices.
        # (num_box_preds, fut_mode, fut_ts)
        dist = torch.linalg.norm(cum_traj_targets[:, None, :, :] - cum_traj_preds, dim=-1)
        dist = dist * gt_fut_masks[:, None, :]
        dist = dist[..., -1]
        dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0
        min_mode_idxs = torch.argmin(dist, dim=-1).tolist()
        box_idxs = torch.arange(traj_preds.shape[0]).tolist()
        best_traj_preds = traj_preds[box_idxs, min_mode_idxs, :, :].reshape(-1, self.fut_ts*2)

        return best_traj_preds

    def get_traj_cls_target(self,
             traj_preds,
             traj_targets,
             gt_fut_masks,
             neg_inds):
        """"Get Trajectory mode classification target.
        Args:
            traj_preds (Tensor): MultiModal traj preds with shape (num_box_preds, fut_mode, fut_ts, 2).
            traj_targets (Tensor): Ground truth traj for each pred box with shape (num_box_preds, fut_ts, 2).
            gt_fut_masks (Tensor): Ground truth traj mask with shape (num_box_preds, fut_ts).
            neg_inds (Tensor): Negtive indices with shape (num_box_preds,)

        Returns:
            traj_labels (Tensor): traj cls labels (num_box_preds,).
        """

        cum_traj_preds = traj_preds.cumsum(dim=-2)
        cum_traj_targets = traj_targets.cumsum(dim=-2)

        # Get min pred mode indices.
        # (num_box_preds, fut_mode, fut_ts)
        dist = torch.linalg.norm(cum_traj_targets[:, None, :, :] - cum_traj_preds, dim=-1)
        dist = dist * gt_fut_masks[:, None, :]
        dist = dist[..., -1]
        dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0
        traj_labels = torch.argmin(dist, dim=-1)
        traj_labels[neg_inds] = self.fut_mode

        return traj_labels

    def map_loss_single(self,
                    cls_scores,
                    bbox_preds,
                    pts_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_pts_list (list[Tensor]): Ground truth pts for each image
                with shape (num_gts, fixed_num, 2) in [x,y] format.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.map_get_targets(cls_scores_list, bbox_preds_list,pts_preds_list,
                                           gt_bboxes_list, gt_labels_list,gt_shifts_pts_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
 
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.map_cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.map_bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_map_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
        # normalized_bbox_targets = bbox_targets
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.map_code_weights

        loss_bbox = self.loss_map_bbox(
            bbox_preds[isnotnan, :4],
            normalized_bbox_targets[isnotnan,:4],
            bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos)

        # regression pts CD loss
        # num_samples, num_order, num_pts, num_coords
        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)

        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))
        if self.map_num_pts_per_vec != self.map_num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0,2,1)
            pts_preds = F.interpolate(pts_preds, size=(self.map_num_pts_per_gt_vec), mode='linear',
                                    align_corners=True)
            pts_preds = pts_preds.permute(0,2,1).contiguous()

        loss_pts = self.loss_map_pts(
            pts_preds[isnotnan,:,:],
            normalized_pts_targets[isnotnan,:,:],
            pts_weights[isnotnan,:,:],
            avg_factor=num_total_pos)

        loss_curvature = self.loss_map_curvature(
            pts_preds[isnotnan,:,:],
            normalized_pts_targets[isnotnan,:,:],
            avg_factor=num_total_pos)

        dir_weights = pts_weights[:, :-self.map_dir_interval,0]
        denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        denormed_pts_preds_dir = denormed_pts_preds[:,self.map_dir_interval:,:] - \
            denormed_pts_preds[:,:-self.map_dir_interval,:]
        pts_targets_dir = pts_targets[:, self.map_dir_interval:,:] - pts_targets[:,:-self.map_dir_interval,:]

        loss_dir = self.loss_map_dir(
            denormed_pts_preds_dir[isnotnan,:,:],
            pts_targets_dir[isnotnan,:,:],
            dir_weights[isnotnan,:],
            avg_factor=num_total_pos)

        bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_map_iou(
            bboxes[isnotnan, :4],
            bbox_targets[isnotnan, :4],
            bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos)

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_iou = torch.nan_to_num(loss_iou)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)
            loss_curvature = torch.nan_to_num(loss_curvature)

        return loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir, loss_curvature

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             map_gt_bboxes_list,
             map_gt_labels_list,
             preds_dicts,
             ego_fut_gt,
             ego_fut_masks,
             ego_fut_cmd,
             gt_attr_labels,
             gt_bboxes_ignore=None,
             map_gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        map_gt_vecs_list = copy.deepcopy(map_gt_bboxes_list)

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        all_traj_preds = preds_dicts['all_traj_preds']
        all_traj_cls_scores = preds_dicts['all_traj_cls_scores']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        map_all_cls_scores = preds_dicts['map_all_cls_scores']
        map_all_bbox_preds = preds_dicts['map_all_bbox_preds']
        map_all_pts_preds = preds_dicts['map_all_pts_preds']
        map_enc_cls_scores = preds_dicts['map_enc_cls_scores']
        map_enc_bbox_preds = preds_dicts['map_enc_bbox_preds']
        map_enc_pts_preds = preds_dicts['map_enc_pts_preds']
        ego_fut_preds = preds_dicts['ego_fut_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        # Infra unmatched queries sit at the tail of the combined query pool.
        # Excluding them from the fusion loss prevents Hungarian matching from
        # assigning background labels to out-of-FOV infra queries, which would
        # suppress their confidence over training epochs.
        n_infra_unmatched = preds_dicts.get('n_infra_unmatched', 0)
        all_n_infra_list = [n_infra_unmatched] * num_dec_layers

        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_attr_labels_list = [gt_attr_labels for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox, loss_traj, loss_traj_cls = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds, all_traj_preds,
            all_traj_cls_scores, all_gt_bboxes_list, all_gt_labels_list,
            all_gt_attr_labels_list, all_gt_bboxes_ignore_list, all_n_infra_list)
        

        num_dec_layers = len(map_all_cls_scores)
        device = map_gt_labels_list[0].device

        map_gt_bboxes_list = [
            map_gt_bboxes.bbox.to(device) for map_gt_bboxes in map_gt_vecs_list]
        map_gt_pts_list = [
            map_gt_bboxes.fixed_num_sampled_points.to(device) for map_gt_bboxes in map_gt_vecs_list]
        if self.map_gt_shift_pts_pattern == 'v0':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v1':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v1.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v2':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v2.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v3':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v3.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v4':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v4.to(device) for gt_bboxes in map_gt_vecs_list]
        else:
            raise NotImplementedError
        map_all_gt_bboxes_list = [map_gt_bboxes_list for _ in range(num_dec_layers)]
        map_all_gt_labels_list = [map_gt_labels_list for _ in range(num_dec_layers)]
        map_all_gt_pts_list = [map_gt_pts_list for _ in range(num_dec_layers)]
        map_all_gt_shifts_pts_list = [map_gt_shifts_pts_list for _ in range(num_dec_layers)]
        map_all_gt_bboxes_ignore_list = [
            map_gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        map_losses_cls, map_losses_bbox, map_losses_iou, \
            map_losses_pts, map_losses_dir, map_losses_curvature = multi_apply(
            self.map_loss_single, map_all_cls_scores, map_all_bbox_preds,
            map_all_pts_preds, map_all_gt_bboxes_list, map_all_gt_labels_list,
            map_all_gt_shifts_pts_list, map_all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_traj'] = loss_traj[-1]
        loss_dict['loss_traj_cls'] = loss_traj_cls[-1]
        # loss from the last decoder layer
        loss_dict['loss_map_cls'] = map_losses_cls[-1]
        loss_dict['loss_map_bbox'] = map_losses_bbox[-1]
        loss_dict['loss_map_iou'] = map_losses_iou[-1]
        loss_dict['loss_map_pts'] = map_losses_pts[-1]
        loss_dict['loss_map_dir'] = map_losses_dir[-1]
        loss_dict['loss_map_curvature'] = map_losses_curvature[-1]

        # Planning Loss
        ego_fut_gt = ego_fut_gt.squeeze(1)
        ego_fut_masks = ego_fut_masks.squeeze(1).squeeze(1)
        ego_fut_cmd = ego_fut_cmd.squeeze(1).squeeze(1)

        batch, num_agent = all_traj_preds[-1].shape[:2]
        agent_fut_preds = all_traj_preds[-1].view(batch, num_agent, self.fut_mode, self.fut_ts, 2)
        agent_fut_cls_preds = all_traj_cls_scores[-1].view(batch, num_agent, self.fut_mode)
        loss_plan_input = [ego_fut_preds, ego_fut_gt, ego_fut_masks, ego_fut_cmd,
                           map_all_pts_preds[-1], map_all_cls_scores[-1].sigmoid(),
                           all_bbox_preds[-1][..., 0:2], agent_fut_preds,
                           all_cls_scores[-1].sigmoid(), agent_fut_cls_preds.sigmoid()]

        loss_planning_dict = self.loss_planning(*loss_plan_input)
        loss_dict['loss_plan_reg'] = loss_planning_dict['loss_plan_reg']
        loss_dict['loss_plan_bound'] = loss_planning_dict['loss_plan_bound']
        loss_dict['loss_plan_col'] = loss_planning_dict['loss_plan_col']
        loss_dict['loss_plan_dir'] = loss_planning_dict['loss_plan_dir']

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        # loss from other decoder layers
        num_dec_layer = 0
        for map_loss_cls_i, map_loss_bbox_i, map_loss_iou_i, map_loss_pts_i, map_loss_dir_i, map_loss_curv_i in zip(
            map_losses_cls[:-1],
            map_losses_bbox[:-1],
            map_losses_iou[:-1],
            map_losses_pts[:-1],
            map_losses_dir[:-1],
            map_losses_curvature[:-1]
        ):
            loss_dict[f'd{num_dec_layer}.loss_map_cls'] = map_loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_map_bbox'] = map_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_map_iou'] = map_loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_map_pts'] = map_loss_pts_i
            loss_dict[f'd{num_dec_layer}.loss_map_dir'] = map_loss_dir_i
            loss_dict[f'd{num_dec_layer}.loss_map_curvature'] = map_loss_curv_i
            num_dec_layer += 1

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list,
                                 gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        if map_enc_cls_scores is not None:
            map_binary_labels_list = [
                torch.zeros_like(map_gt_labels_list[i])
                for i in range(len(map_all_gt_labels_list))
            ]
            # TODO bug here, but we dont care enc_loss now
            map_enc_loss_cls, map_enc_loss_bbox, map_enc_loss_iou, \
                 map_enc_loss_pts, map_enc_loss_dir, map_enc_loss_curv = \
                self.map_loss_single(
                    map_enc_cls_scores, map_enc_bbox_preds,
                    map_enc_pts_preds, map_gt_bboxes_list,
                    map_binary_labels_list, map_gt_pts_list,
                    map_gt_bboxes_ignore
                )
            loss_dict['enc_loss_map_cls'] = map_enc_loss_cls
            loss_dict['enc_loss_map_bbox'] = map_enc_loss_bbox
            loss_dict['enc_loss_map_iou'] = map_enc_loss_iou
            loss_dict['enc_loss_map_pts'] = map_enc_loss_pts
            loss_dict['enc_loss_map_dir'] = map_enc_loss_dir
            loss_dict['enc_loss_map_curvature'] = map_enc_loss_curv        
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        det_preds_dicts = self.bbox_coder.decode(preds_dicts)
        # map_bboxes: xmin, ymin, xmax, ymax
        map_preds_dicts = self.map_bbox_coder.decode(preds_dicts)

        num_samples = len(det_preds_dicts)
        assert len(det_preds_dicts) == len(map_preds_dicts), \
             'len(preds_dict) should be equal to len(map_preds_dicts)'
        ret_list = []
        for i in range(num_samples):
            preds = det_preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']
            trajs = preds['trajs']

            map_preds = map_preds_dicts[i]
            map_bboxes = map_preds['map_bboxes']
            map_scores = map_preds['map_scores']
            map_labels = map_preds['map_labels']
            map_pts = map_preds['map_pts']

            ret_list.append([bboxes, scores, labels, trajs, map_bboxes,
                             map_scores, map_labels, map_pts])

        return ret_list

    def select_and_pad_pred_map(
        self,
        motion_pos,
        map_query,
        map_score,
        map_pos,
        map_thresh=0.5,
        dis_thresh=None,
        pe_normalization=True,
        use_fix_pad=False
    ):
        """select_and_pad_pred_map.
        Args:
            motion_pos: [B, A, 2]
            map_query: [B, P, D].
            map_score: [B, P, 3].
            map_pos: [B, P, pts, 2].
            map_thresh: map confidence threshold for filtering low-confidence preds
            dis_thresh: distance threshold for masking far maps for each agent in cross-attn
            use_fix_pad: always pad one lane instance for each batch
        Returns:
            selected_map_query: [B*A, P1(+1), D], P1 is the max inst num after filter and pad.
            selected_map_pos: [B*A, P1(+1), 2] [B,P1,2] -> [1,1,2]
            selected_padding_mask: [B*A, P1(+1)]
        """
        
        if dis_thresh is None:
            raise NotImplementedError('Not implement yet')

        # use the most close pts pos in each map inst as the inst's pos
        batch, num_map = map_pos.shape[:2]
        map_dis = torch.sqrt(map_pos[..., 0]**2 + map_pos[..., 1]**2)
        min_map_pos_idx = map_dis.argmin(dim=-1).flatten()  # [B*P]
        min_map_pos = map_pos.flatten(0, 1)  # [B*P, pts, 2]
        min_map_pos = min_map_pos[range(min_map_pos.shape[0]), min_map_pos_idx]  # [B*P, 2]
        min_map_pos = min_map_pos.view(batch, num_map, 2)  # [B, P, 2]

        # select & pad map vectors for different batch using map_thresh
        map_score = map_score.sigmoid()
        map_max_score = map_score.max(dim=-1)[0]
        map_idx = map_max_score > map_thresh
        batch_max_pnum = 0
        for i in range(map_score.shape[0]):
            pnum = map_idx[i].sum()
            if pnum > batch_max_pnum:
                batch_max_pnum = pnum

        selected_map_query, selected_map_pos, selected_padding_mask = [], [], []
        for i in range(map_score.shape[0]):
            dim = map_query.shape[-1]
            valid_pnum = map_idx[i].sum()
            valid_map_query = map_query[i, map_idx[i]]
            valid_map_pos = min_map_pos[i, map_idx[i]]
            pad_pnum = batch_max_pnum - valid_pnum
            padding_mask = torch.tensor([False], device=map_score.device).repeat(batch_max_pnum)
            if pad_pnum != 0:
                valid_map_query = torch.cat([valid_map_query, torch.zeros((pad_pnum, dim), device=map_score.device)], dim=0)
                valid_map_pos = torch.cat([valid_map_pos, torch.zeros((pad_pnum, 2), device=map_score.device)], dim=0)
                padding_mask[valid_pnum:] = True
            selected_map_query.append(valid_map_query)
            selected_map_pos.append(valid_map_pos)
            selected_padding_mask.append(padding_mask)

        selected_map_query = torch.stack(selected_map_query, dim=0)
        selected_map_pos = torch.stack(selected_map_pos, dim=0)
        selected_padding_mask = torch.stack(selected_padding_mask, dim=0)

        # generate different pe for map vectors for each agent
        num_agent = motion_pos.shape[1]
        selected_map_query = selected_map_query.unsqueeze(1).repeat(1, num_agent, 1, 1)  # [B, A, max_P, D]
        selected_map_pos = selected_map_pos.unsqueeze(1).repeat(1, num_agent, 1, 1)  # [B, A, max_P, 2]
        selected_padding_mask = selected_padding_mask.unsqueeze(1).repeat(1, num_agent, 1)  # [B, A, max_P]
        # move lane to per-car coords system
        selected_map_dist = selected_map_pos - motion_pos[:, :, None, :]  # [B, A, max_P, 2]
        if pe_normalization:
            selected_map_pos = selected_map_pos - motion_pos[:, :, None, :]  # [B, A, max_P, 2]

        # filter far map inst for each agent
        map_dis = torch.sqrt(selected_map_dist[..., 0]**2 + selected_map_dist[..., 1]**2)
        valid_map_inst = (map_dis <= dis_thresh)  # [B, A, max_P]
        invalid_map_inst = (valid_map_inst == False)
        selected_padding_mask = selected_padding_mask + invalid_map_inst

        selected_map_query = selected_map_query.flatten(0, 1)
        selected_map_pos = selected_map_pos.flatten(0, 1)
        selected_padding_mask = selected_padding_mask.flatten(0, 1)

        num_batch = selected_padding_mask.shape[0]
        feat_dim = selected_map_query.shape[-1]
        if use_fix_pad:
            pad_map_query = torch.zeros((num_batch, 1, feat_dim), device=selected_map_query.device)
            pad_map_pos = torch.ones((num_batch, 1, 2), device=selected_map_pos.device)
            pad_lane_mask = torch.tensor([False], device=selected_padding_mask.device).unsqueeze(0).repeat(num_batch, 1)
            selected_map_query = torch.cat([selected_map_query, pad_map_query], dim=1)
            selected_map_pos = torch.cat([selected_map_pos, pad_map_pos], dim=1)
            selected_padding_mask = torch.cat([selected_padding_mask, pad_lane_mask], dim=1)

        return selected_map_query, selected_map_pos, selected_padding_mask


    def select_and_pad_query(
        self,
        query,
        query_pos,
        query_score,
        score_thresh=0.5,
        use_fix_pad=True
    ):
        """select_and_pad_query.
        Args:
            query: [B, Q, D].
            query_pos: [B, Q, 2]
            query_score: [B, Q, C].
            score_thresh: confidence threshold for filtering low-confidence query
            use_fix_pad: always pad one query instance for each batch
        Returns:
            selected_query: [B, Q', D]
            selected_query_pos: [B, Q', 2]
            selected_padding_mask: [B, Q']
        """

        # select & pad query for different batch using score_thresh
        query_score = query_score.sigmoid()
        query_score = query_score.max(dim=-1)[0]
        query_idx = query_score > score_thresh
        batch_max_qnum = 0
        for i in range(query_score.shape[0]):
            qnum = query_idx[i].sum()
            if qnum > batch_max_qnum:
                batch_max_qnum = qnum

        selected_query, selected_query_pos, selected_padding_mask = [], [], []
        for i in range(query_score.shape[0]):
            dim = query.shape[-1]
            valid_qnum = query_idx[i].sum()
            valid_query = query[i, query_idx[i]]
            valid_query_pos = query_pos[i, query_idx[i]]
            pad_qnum = batch_max_qnum - valid_qnum
            padding_mask = torch.tensor([False], device=query_score.device).repeat(batch_max_qnum)
            if pad_qnum != 0:
                valid_query = torch.cat([valid_query, torch.zeros((pad_qnum, dim), device=query_score.device)], dim=0)
                valid_query_pos = torch.cat([valid_query_pos, torch.zeros((pad_qnum, 2), device=query_score.device)], dim=0)
                padding_mask[valid_qnum:] = True
            selected_query.append(valid_query)
            selected_query_pos.append(valid_query_pos)
            selected_padding_mask.append(padding_mask)

        selected_query = torch.stack(selected_query, dim=0)
        selected_query_pos = torch.stack(selected_query_pos, dim=0)
        selected_padding_mask = torch.stack(selected_padding_mask, dim=0)

        num_batch = selected_padding_mask.shape[0]
        feat_dim = selected_query.shape[-1]
        if use_fix_pad:
            pad_query = torch.zeros((num_batch, 1, feat_dim), device=selected_query.device)
            pad_query_pos = torch.ones((num_batch, 1, 2), device=selected_query_pos.device)
            pad_mask = torch.tensor([False], device=selected_padding_mask.device).unsqueeze(0).repeat(num_batch, 1)
            selected_query = torch.cat([selected_query, pad_query], dim=1)
            selected_query_pos = torch.cat([selected_query_pos, pad_query_pos], dim=1)
            selected_padding_mask = torch.cat([selected_padding_mask, pad_mask], dim=1)
        
        return selected_query, selected_query_pos, selected_padding_mask

    def select_and_mask_map(
    self,
    motion_pos: torch.Tensor,   # [B, A, 2]
    map_query: torch.Tensor,    # [B, P, D]
    map_score: torch.Tensor,    # [B, P, C] (logits)
    map_pos_pts: torch.Tensor,  # [B, P, pts, 2]
    *,
    map_thresh: float = 0.5,
    dis_thresh: float = None,
    road_edge_idx: int = None,
    pe_normalization: bool = False,
    use_fix_pad: bool = False,):
        """
        Returns (like select_and_pad_pred_map):
            selected_map_query: [B*A, P1(+1), D]
            selected_map_pos:   [B*A, P1(+1), 2]   (relative to agent if pe_normalization=True)
            selected_padding_mask: [B*A, P1(+1)]   (True = padded/invalid)
        """
        if dis_thresh is None:
            raise NotImplementedError('dis_thresh must be provided')

        B, P, pts = map_pos_pts.shape[:3]
        device = map_query.device
        D = map_query.shape[-1]

        # 1) Reduce per-instance pts -> single [x,y] (closest point to origin)
        map_dis = torch.sqrt(map_pos_pts[..., 0]**2 + map_pos_pts[..., 1]**2)    # [B,P,pts]
        min_idx = map_dis.argmin(dim=-1).flatten()                                # [B*P]
        flat_pts = map_pos_pts.flatten(0, 1)                                      # [B*P, pts, 2]
        min_map_pos = flat_pts[torch.arange(flat_pts.shape[0], device=device), min_idx]  # [B*P,2]
        min_map_pos = min_map_pos.view(B, P, 2)                                   # [B,P,2]

        # 2) Confidence + class filter (pre-padding, per batch)
        prob = map_score.sigmoid().max(dim=-1)[0]      # [B,P]
        keep = prob > map_thresh                       # [B,P]
        
        if road_edge_idx is not None:
            top1 = map_score.argmax(dim=-1)            # [B,P]
            keep = keep & (top1 != road_edge_idx)

        # 3) Pad to batch_max_pnum (same as your select_and_pad_pred_map)
        batch_max_pnum = int(map_query.shape[1]) if B > 0 else 0
        sel_Q, sel_pos, sel_mask = [], [], []
        for b in range(B):
            kb = keep[b]
            if not keep[b].any():
                # All False -> randomly activate one (or more) elements
                rand_idx = torch.randperm(P, device=keep.device)[:int(P)]
                keep[b, rand_idx] = True

            q_b   = map_query[b, kb]           # [Pb, D]
            pos_b = min_map_pos[b, kb]         # [Pb, 2]
            Pb = q_b.shape[0]
            pad_p = batch_max_pnum - Pb
            padding_mask = torch.zeros(batch_max_pnum, dtype=torch.bool, device=device)
            if pad_p > 0:
                q_b   = torch.cat([q_b,   torch.zeros(pad_p, D, device=device)], dim=0)
                pos_b = torch.cat([pos_b, torch.zeros(pad_p, 2, device=device)], dim=0)
                padding_mask[Pb:] = True
            sel_Q.append(q_b)                  # [P1, D]
            sel_pos.append(pos_b)              # [P1, 2]
            sel_mask.append(padding_mask)      # [P1]

        # B-only tensors (before radius)
        b_map_query = torch.stack(sel_Q,   dim=0)   # [B, P1, D]
        b_map_pos   = torch.stack(sel_pos, dim=0)   # [B, P1, 2]
        b_pad_mask  = torch.stack(sel_mask,dim=0)   # [B, P1]

        # ---- B-only STRICT radius mask across agents ----
        # dist_all[b, a, p] = || b_map_pos[b,p] - motion_pos[b,a] ||
        rel_all  = b_map_pos[:, None, :, :] - motion_pos[:, :, None, :]  # [B, A, P1, 2]
        dist_all = torch.linalg.norm(rel_all, dim=-1)                     # [B, A, P1]
        #valid_b  = (dist_all >= dis_thresh).all(dim=1)                    # [B, P1]  (ALL-far)
        valid_b = (dist_all.mean(dim=1) > dis_thresh)
        b_pad_mask = b_pad_mask | (~valid_b)                              # True = invalid

        # Optional: positional normalization for B-only
        if pe_normalization:
            # Use nearest agent for stable relative coords
            near_idx = dist_all.argmin(dim=1)                             # [B, P1]
            b_idx = torch.arange(B, device=device)[:, None].expand(B, b_map_pos.size(1))
            near_agent_pos = motion_pos[b_idx, near_idx, :]               # [B, P1, 2]
            b_map_pos = b_map_pos - near_agent_pos                        # [B, P1, 2]

        # 4) Expand per agent (BA branch) and per-agent radius mask (unchanged)
        A = motion_pos.shape[1]
        sel_Q_ba   = b_map_query.unsqueeze(1).repeat(1, A, 1, 1)  # [B,A,P1,D]
        sel_pos_ba = b_map_pos.unsqueeze(1).repeat(1, A, 1, 1)    # [B,A,P1,2]
        sel_mask_ba = b_pad_mask.unsqueeze(1).repeat(1, A, 1)     # start from B-only mask so both agree

        rel = sel_pos_ba - motion_pos[:, :, None, :]              # [B,A,P1,2]
        if pe_normalization:
            sel_pos_ba = rel
        dist = torch.linalg.norm(rel, dim=-1)                     # [B,A,P1]
        valid = (dist >= dis_thresh)                              # keep outside per-agent
        sel_mask_ba = sel_mask_ba | (~valid)
        
        # 5) Flatten (B,A) -> [B*A, ...]
        ba_map_query = sel_Q_ba.flatten(0, 1)                     # [B*A, P1, D]
        ba_map_pos   = sel_pos_ba.flatten(0, 1)                   # [B*A, P1, 2]
        ba_pad_mask  = sel_mask_ba.flatten(0, 1)                  # [B*A, P1]

        # Safety: if every key is masked for an agent, unmask the first slot so
        # that softmax receives at least one finite entry and does not produce NaN.
        all_masked_ba = ba_pad_mask.all(dim=-1)        # [B*A]
        if all_masked_ba.any():
            ba_pad_mask[all_masked_ba, 0] = False

        all_masked_b = b_pad_mask.all(dim=-1)          # [B]
        if all_masked_b.any():
            b_pad_mask[all_masked_b, 0] = False

        return ba_map_query, ba_map_pos, ba_pad_mask, b_map_query, b_map_pos, b_pad_mask
