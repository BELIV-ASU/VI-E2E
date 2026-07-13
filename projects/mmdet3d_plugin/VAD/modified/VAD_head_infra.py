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

#torch.autograd.set_detect_anomaly(True)

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
class VADHeadInfra(DETRHead):
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
                 agent_decoder = None,
                 agent_map_decoder = None,
                 map_agent_decoder = None,
                 **kwargs):

        self.vi_agent_fuser = vi_agent_fuser
        self.vi_map_fuser = vi_map_fuser
        self.vi_motion = vi_motion
        self.vi_map_motion = vi_map_motion

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
        if map_class_weight is not None and (self.__class__ is VADHeadInfra):
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

        super(VADHeadInfra, self).__init__(*args, transformer=transformer, **kwargs)
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
        self.loss_plan_reg = build_loss(loss_plan_reg)
        self.loss_plan_bound = build_loss(loss_plan_bound)
        self.loss_plan_col = build_loss(loss_plan_col)
        self.loss_plan_dir = build_loss(loss_plan_dir)

        with open('/scratch/jmeng18/V2X-Seq-SPD-New/infrastructure-side/v1.0-trainval/ego_pose.json','r') as f:
            self.ego_pose_infra = json.load(f)
        
        self.map_transform_mlp = nn.Linear(16, self.embed_dims) 
        self.agent_transform_mlp = nn.Linear(16, self.embed_dims)
        self.norm_map_pos = nn.LayerNorm(self.embed_dims)
        self.norm_agent_pos = nn.LayerNorm(self.embed_dims)

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


        map_reg_branch = []
        for _ in range(self.num_reg_fcs):
            map_reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            map_reg_branch.append(nn.ReLU())
        map_reg_branch.append(Linear(self.embed_dims, self.map_code_size))
        map_reg_branch = nn.Sequential(*map_reg_branch)



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
            self.map_reg_branches = _get_clones(map_reg_branch, map_num_pred)
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
            self.map_reg_branches = nn.ModuleList(
                [map_reg_branch for _ in range(map_num_pred)])

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

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
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

    # Modifications: Generate the map, agent, ego queries
    def query_generation(self, outputs):
        bev_embed, hs, init_reference, inter_references, \
            map_hs, map_init_reference, map_inter_references = outputs

        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
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
            tmp[..., 4:5] = tmp[..., 4:5] + reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            infra_tmp = tmp.clone()
            infra_class = outputs_class.clone()
            
            outputs_coords_bev.append(tmp[..., 0:2].clone().detach())

            outputs_classes.append(outputs_class)
            
        # Map elements:
        for lvl in range(map_hs.shape[0]):
            if lvl == 0:
                reference = map_init_reference
            else:
                reference = map_inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            map_outputs_class = self.map_cls_branches[lvl](
                map_hs[lvl].view(1,self.map_num_vec, self.map_num_pts_per_vec,-1).mean(2)
            )
            tmp = self.map_reg_branches[lvl](map_hs[lvl])
            # TODO: check the shape of reference
            assert reference.shape[-1] == 2
            #tmp = tmp.clone()
            #tmp[..., 0:2] = tmp[..., 0:2]+reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2]+reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid() # cx,cy,w,h
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
        ego_his_feats, infra_tmp, infra_class

    def get_transform(self,ego_pose, token):
        for entry in ego_pose:
            if entry["token"] == token:
                rot = R.from_quat(entry["rotation"]).as_matrix()  # [3, 3]
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

    # @auto_fp16(apply_to=('mlvl_feats'))
    @force_fp32(apply_to=('mlvl_feats', 'prev_bev'))
    def forward(self,
                mlvl_feats,
                img_metas,
                prev_bev=None,
                only_bev=False,
                ego_his_trajs=None,
                ego_lcf_feat=None,
                training_mode = None,
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
           
            infra_feats = mlvl_feats
            # import pdb; pdb.set_trace()
            if training_mode:
                with torch.no_grad():
                    outputs_i = self.transformer(
                        infra_feats,
                        bev_queries,
                        object_query_embeds,
                        map_query_embeds,
                        self.bev_h,
                        self.bev_w,
                        grid_length=(self.real_h / self.bev_h,
                                     self.real_w / self.bev_w),
                        bev_pos=bev_pos, #DEBUG: changed from bevA_pos to None
                        reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                        cls_branches=self.cls_branches if self.as_two_stage else None,
                        map_reg_branches=self.map_reg_branches if self.with_box_refine else None,  # noqa:E501
                        map_cls_branches=self.map_cls_branches if self.as_two_stage else None,
                        img_metas=img_metas,
                        prev_bev=prev_bev
                    )
            else:
                outputs_i = self.transformer(
                    infra_feats,
                    bev_queries,
                    object_query_embeds,
                    map_query_embeds,
                    self.bev_h,
                    self.bev_w,
                    grid_length=(self.real_h / self.bev_h,
                                 self.real_w / self.bev_w),
                    bev_pos=bev_pos, #DEBUG: changed from bevA_pos to None
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
        
        bev_embed_i, hs_i, init_reference_i, inter_references_i, \
            map_hs_i, map_init_reference_i, map_inter_references_i = outputs_i

        #prepare transformation matrices from file
        sample_token_inf = img_metas[0]['filename'][1].split('/')[-1][:-4]

        T_inf = self.get_transform(self.ego_pose_infra, sample_token_inf)
        
        #get query from modality
        
        #infra
        ego_query_i, agent_query_i, ego_pos_emb_i, agent_pos_emb_i, agent_mask_i, map_query_i, map_pos_emb_i, map_mask_i,\
        ego_his_feats_i, infra_tmp, infra_class = self.query_generation(outputs_i)

        #Spatial synchronization

        #infra global coordinate embedding

        T_inf_tensor = torch.tensor(T_inf, dtype=map_query_i.dtype, device=map_query_i.device).view(1, 1, -1)  # [1, 1, 16]
        T_inf_embed_map = self.map_transform_mlp(T_inf_tensor.expand(map_query_i.shape[0], map_query_i.shape[1], -1))    # [B, M, D]
        T_inf_embed_agent = self.agent_transform_mlp(T_inf_tensor.expand(agent_query_i.shape[0], agent_query_i.shape[1], -1))  # [B, N, D]

        map_pos_emb_i = self.norm_map_pos(map_pos_emb_i.clone() + T_inf_embed_map)
        map_query_i = map_query_i.clone() + map_pos_emb_i
        infra_map_pos_embed = map_pos_emb_i

        agent_pos_emb_i = self.norm_agent_pos(agent_pos_emb_i + T_inf_embed_agent)
        agent_query_i = agent_query_i.clone() + agent_pos_emb_i
        infra_agent_pos_embed = agent_pos_emb_i

        
        
        # return ego_query_i, agent_query_i, ego_pos_emb_i, agent_pos_emb_i, agent_mask_i, map_query_i, map_pos_emb_i, map_mask_i,\
        # ego_his_feats_i

        agent_query_i = agent_query_i.permute(1,0,2)

        # # Do classification and regression of map/agent classes ans coordinates from fused queries
        outputs_classes = []
        outputs_coords_bev = []

        #outputs_class = self.cls_branches[-1](agent_query_i.permute(1,0,2))#self.cls_branches(agent_query)
        #tmp = self.reg_branches[-1](agent_query_i.permute(1,0,2))
        #reference = inverse_sigmoid(inter_references_i[-1])
        inf_pc_range = [0, -51.2, -5.0, 102.4, 51.2, 3.0]
        infra_reference = torch.cat([infra_tmp[...,:2], infra_tmp[...,4:5]], dim=2)
        tmp = infra_tmp
        outputs_class = infra_class
        #reference = inter_references_i[-1]
        #reference = inverse_sigmoid(reference)
        
        
        #tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2]
        #tmp[..., 4:5] = tmp[..., 4:5] + reference[..., 2:3]

        #tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
        #tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
        tmp[..., 0:1] = (infra_reference[..., 0:1] * (inf_pc_range[3] - inf_pc_range[0]) + inf_pc_range[0])
        tmp[..., 1:2] = (infra_reference[..., 1:2] * (inf_pc_range[4] - inf_pc_range[1]) + inf_pc_range[1])
        tmp[..., 4:5] = (infra_reference[..., 2:3] * (inf_pc_range[5] - inf_pc_range[2]) + inf_pc_range[2])

        #xy = tmp[0][...,:2].cpu().detach().numpy()
        #plt.figure(figsize=(6,6))
        #plt.scatter(xy[:, 0], xy[:, 1], s=5)   # s = point size
        #plt.xlabel("X")
        #plt.ylabel("Y")
        #plt.title("2D Scatter Plot of 900 Points")
        #plt.axis("equal")    # keep aspect ratio
        #plt.grid(True)
        #plt.show()

        
        # Convert the infra_coords to global coords
        
        infra_coords = torch.cat([tmp[..., 0:2], tmp[...,4:5]], dim=2).squeeze(0).clone()
        ones = torch.ones((infra_coords.shape[0], 1), device=infra_coords.device)
        infra_h = torch.cat([infra_coords, ones], dim=1)
        global_h = infra_h.cpu().detach() @ T_inf.T
        infra_global_coords = global_h[:, :3]
        #import pdb; pdb.set_trace()
        #tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
        outputs_coords_bev.append(tmp[..., 0:2].clone().detach())
        outputs_classes.append(outputs_class)
        # # VI motion queries
        # # motion SA

        motion_query = self.vi_motion(
            query=agent_query_i.permute(1, 0, 2),
            key=agent_query_i.permute(1, 0, 2),
            value=agent_query_i.permute(1, 0, 2),
            query_pos=None,#veh_agent_pos_embed,
            key_pos=None,#veh_agent_pos_embed,
            key_padding_mask=agent_mask_i.bool()
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

        # if self.motion_det_score is not None:
        #     motion_score = outputs_classes[-1]
        #     max_motion_score = motion_score.max(dim=-1)[0]
        #     invalid_motion_idx = max_motion_score < self.motion_det_score  # [B, A]
        #     invalid_motion_idx = invalid_motion_idx.unsqueeze(2).repeat(1, 1, self.fut_mode).flatten(1, 2)
        # else:
        #     invalid_motion_idx = None


        ca_motion_query = motion_query.permute(1, 0, 2).flatten(0, 1).unsqueeze(0) #[B,M,D]
        
        return ego_query_i, agent_query_i, ego_pos_emb_i, agent_pos_emb_i, agent_mask_i, map_query_i, map_pos_emb_i, map_mask_i,\
        ego_his_feats_i, ca_motion_query, infra_global_coords, outputs_classes, infra_coords, infra_tmp
        

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


    
