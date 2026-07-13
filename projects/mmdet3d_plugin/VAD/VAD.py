import time
import copy

import torch
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmcv.runner import force_fp32, auto_fp16
from scipy.optimize import linear_sum_assignment
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.VAD.planner.metric_stp3 import PlanningMetric

from mmdet3d.models.builder import build_head, build_backbone, build_neck
try:
    import wandb
except Exception:
    wandb = None
import pickle
import numpy as np
from pyquaternion import Quaternion

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes




@DETECTORS.register_module()
class VAD(MVXTwoStageDetector):
    """VAD model.
    """
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 img_backbone_infra=None,
                 pts_backbone=None,
                 img_neck=None,
                 img_neck_infra=None,
                 pts_neck=None,
                 freeze_feature_extraction=False,
                 num_trainable_map_layers=2,
                 pts_bbox_head=None,
                 pts_bbox_head_infra=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 training_mode=None,
                 infra_loss_weight=0.05,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 fut_ts=6,
                 fut_mode=6
                 ):

        super(VAD,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)

        
        self.img_backbone_infra = (
            build_backbone(img_backbone_infra) if img_backbone_infra is not None else None
        )
        self.img_neck_infra = (
            build_neck(img_neck_infra) if img_neck_infra is not None else None
        )
        self.with_img_neck_infra = self.img_neck_infra is not None

        if pts_bbox_head_infra is not None:
            # Use the infra head's own pc_range instead of reusing the vehicle one.
            infra_det_pc_range = pts_bbox_head_infra.get('bbox_coder', {}).get('pc_range', None)
            infra_map_pc_range = pts_bbox_head_infra.get('map_bbox_coder', {}).get('pc_range', infra_det_pc_range)

            # Copy vehicle train/test cfg first, then rewrite the range-related fields
            # so the infra head is trained in the infra coordinate range.
            pts_train_cfg = copy.deepcopy(train_cfg.pts) if train_cfg else None
            
            if pts_train_cfg is not None:
                if infra_det_pc_range is not None:
                    pts_train_cfg['point_cloud_range'] = infra_det_pc_range
                    if pts_train_cfg.get('assigner', None) is not None:
                        pts_train_cfg['assigner']['pc_range'] = infra_det_pc_range
                if pts_train_cfg.get('map_assigner', None) is not None:
                    if infra_map_pc_range is not None:
                        pts_train_cfg['map_assigner']['pc_range'] = infra_map_pc_range
                    # Infra head is frozen — zero assigner costs so they match
                    # the zeroed loss weights and satisfy the head's assertions.
                    for cost_key in ('cls_cost', 'reg_cost', 'iou_cost', 'pts_cost'):
                        if cost_key in pts_train_cfg['map_assigner']:
                            pts_train_cfg['map_assigner'][cost_key]['weight'] = 0.0
                if pts_train_cfg.get('assigner', None) is not None:
                    for cost_key in ('cls_cost', 'reg_cost', 'iou_cost'):
                        if cost_key in pts_train_cfg['assigner']:
                            pts_train_cfg['assigner'][cost_key]['weight'] = 0.0
            pts_bbox_head_infra.update(train_cfg=pts_train_cfg)

            pts_test_cfg = copy.deepcopy(test_cfg.pts) if test_cfg else None
            if pts_test_cfg is not None and infra_det_pc_range is not None:
                pts_test_cfg['point_cloud_range'] = infra_det_pc_range
            pts_bbox_head_infra.update(test_cfg=pts_test_cfg)

            self.pts_bbox_head_infra = build_head(pts_bbox_head_infra)
            if hasattr(self.pts_bbox_head_infra, 'init_weights'):
                self.pts_bbox_head_infra.init_weights()

        # else:
        #     self.img_backbone_infra = None
        #     self.img_neck_infra = None
        #     self.with_img_neck_infra = False
        #     self.pts_bbox_head_infra = None


        

        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.valid_fut_ts = pts_bbox_head['valid_fut_ts']

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'prev_infra_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

        self.planning_metric = None
        self.training_mode = training_mode
        self.infra_loss_weight = float(infra_loss_weight)

        # Debugging backprop tool
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         def make_hook(name):
        #             def hook(grad):
        #                 print(f"[GradHook] Backward through {name}: grad shape {grad.shape}")
        #             return hook
        #         param.register_hook(make_hook(name))
        self.use_wandb = False

        self.freeze_feature_extraction = freeze_feature_extraction
        self.num_trainable_map_layers = num_trainable_map_layers
        if freeze_feature_extraction:
            self._freeze_feature_extraction()

    def _freeze_feature_extraction(self):
        """Freeze feature extraction, detection, and fusion — only planning/mapping trains."""
        modules_to_freeze = []

        # Backbones and necks
        for attr in ('img_backbone', 'img_backbone_infra', 'img_neck', 'img_neck_infra'):
            m = getattr(self, attr, None)
            if m is not None:
                modules_to_freeze.append(m)

        head = getattr(self, 'pts_bbox_head', None)
        if head is not None:
            # Freeze BEV encoder + detection decoder entirely.
            t = getattr(head, 'transformer', None)
            if t is not None:
                for sub_attr in ('encoder', 'decoder'):
                    sub = getattr(t, sub_attr, None)
                    if sub is not None:
                        modules_to_freeze.append(sub)
                # Freeze all transformer params except the last 2 map decoder layers.
                # The no_grad hooks on frozen map decoder layers break the gradient
                # chain at their outputs, so only the last 2 layers store activations
                # — avoiding OOM without needing torch.utils.checkpoint.
                map_dec = getattr(t, 'map_decoder', None)
                num_trainable_map_layers = getattr(self, 'num_trainable_map_layers', 2)
                trainable_ids = set()
                if map_dec is not None and hasattr(map_dec, 'layers'):
                    for layer in list(map_dec.layers)[-num_trainable_map_layers:]:
                        trainable_ids.update(id(p) for p in layer.parameters())
                for p in t.parameters():
                    if id(p) not in trainable_ids:
                        p.requires_grad = False
                # Freeze the first (total - num_trainable) map decoder layers
                if map_dec is not None and hasattr(map_dec, 'layers'):
                    for layer in list(map_dec.layers)[:-num_trainable_map_layers]:
                        modules_to_freeze.append(layer)

            for attr in ('bev_embedding', 'query_embedding',
                         'cls_branches', 'reg_branches',
                         # map_query_embedding gradient path is broken by the
                         # frozen first-layer no_grad hook, so freeze it too
                         # for consistency (avoids useless grad computation).
                         'map_query_embedding'):
                m = getattr(head, attr, None)
                if m is not None:
                    modules_to_freeze.append(m)
            # map_cls_branches and map_reg_branches stay trainable

            # V2X fusion modules (vehicle + infra query fusion and fused detection heads)
            for attr in ('agent_fusion_decoder', 'cross_agent_fusion',
                         'vi_map_fuser',
                         'bev_embed_linear', 'bev_pos_linear',
                         'cls_branches_fuse', 'reg_branches_fuse',
                         'agent_fus_mlp'):
                m = getattr(head, attr, None)
                if m is not None:
                    modules_to_freeze.append(m)

        # Entire infra detection head
        infra_head = getattr(self, 'pts_bbox_head_infra', None)
        if infra_head is not None:
            modules_to_freeze.append(infra_head)

        def _pre_hook(module, args):
            module._saved_grad_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(False)

        def _post_hook(module, args, output):
            torch.set_grad_enabled(module._saved_grad_enabled)

        for m in modules_to_freeze:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            # Only register once — avoids duplicate hooks on repeated calls
            if not getattr(m, '_no_grad_hook_registered', False):
                m.register_forward_pre_hook(_pre_hook)
                m.register_forward_hook(_post_hook)
                m._no_grad_hook_registered = True

    def train(self, mode=True):
        """Keep frozen modules in eval mode even during training."""
        super().train(mode)
        if getattr(self, 'freeze_feature_extraction', False):
            self._freeze_feature_extraction()
        return self

    def extract_img_feat(self, img, img_metas, len_queue=None, veh = False):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                # img.squeeze_()
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            if veh:
                img_feats = self.img_backbone(img) # vehicle backbone to extract features
            else:
                assert self.img_backbone_infra is not None, "img_backbone_infra is None but infra path requested"
                img_feats = self.img_backbone_infra(img) # infra backbone to extract features

            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            if veh:
                img_feats = self.img_neck(img_feats) # vehicle neck to extract features
            else:
                if self.with_img_neck_infra:
                    img_feats = self.img_neck_infra(img_feats) # infra neck to extract features
                else:
                    # keep features as-is if no infra neck was provided
                    pass

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None, veh = False):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue, veh=veh)
        
        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          map_gt_bboxes_3d,
                          map_gt_labels_3d,                          
                          img_metas,
                          gt_bboxes_ignore=None,
                          map_gt_bboxes_ignore=None,
                          veh_prev_bev=None,
                          infra_prev_bev=None,
                          ego_his_trajs=None,
                          ego_fut_trajs=None,
                          ego_fut_masks=None,
                          ego_fut_cmd=None,
                          ego_lcf_feat=None,
                          gt_attr_labels=None,
                          gt_bboxes_3d_infra=None,
                          gt_labels_3d_infra=None,
                          gt_attr_labels_infra=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        if not(self.training_mode): #vehicle only
            outs = self.pts_bbox_head(pts_feats, img_metas, veh_prev_bev,
                                    ego_his_trajs=ego_his_trajs, ego_lcf_feat=ego_lcf_feat, training_mode = self.training_mode)

        else: #V2X mode
            veh_feats = pts_feats[0]
            infra_feats = pts_feats[1]

            
            # getting queries from infrastructure using infra head
            
            infra_img_metas = copy.deepcopy(img_metas)
            infra_img_metas[0]['filename'] = [img_metas[0]['filename'][1]]
            infra_img_metas[0]['ori_shape'] = [img_metas[0]['ori_shape'][1]]
            infra_img_metas[0]['img_shape'] = [img_metas[0]['img_shape'][1]]
            infra_img_metas[0]['lidar2img'] = [img_metas[0]['lidar2img'][1]]
            infra_img_metas[0]['pts_filename'] = img_metas[0]['filename'][1].replace('/image/', '/velodyne/').rsplit('.', 1)[0] + '.pcd'
            infra_img_metas[0]['can_bus'] = np.array([ 0.        ,  0.        ,  0.        , -0.2616553 , -0.2616553 ,
       -0.2616553 , -0.2616553 ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  3.67106654,  0.        ])
            
            assert infra_prev_bev != None, "infra_prev_bev is None"
            infra_outs = self.pts_bbox_head_infra(infra_feats, infra_img_metas, infra_prev_bev,
                                ego_his_trajs=ego_his_trajs, ego_lcf_feat=ego_lcf_feat, training_mode = self.training_mode)

            infra_queries = infra_outs['queries']
            infra_preds = infra_outs['preds_dicts']
            infra_global_pc_range = infra_outs["global_pc_range"]
            

            outs = self.pts_bbox_head(veh_feats, img_metas, veh_prev_bev,
                                    ego_his_trajs=ego_his_trajs, ego_lcf_feat=ego_lcf_feat, infra_queries = infra_queries, training_mode = self.training_mode, infra_global_pc_range = infra_global_pc_range)

            
        loss_inputs = [
            gt_bboxes_3d, gt_labels_3d, map_gt_bboxes_3d, map_gt_labels_3d,
            outs, ego_fut_trajs, ego_fut_masks, ego_fut_cmd, gt_attr_labels
        ]

        
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)        

        # Compute infrastructure auxiliary loss only when infrastructure GT is provided.
        # Some training pipelines provide only vehicle-frame GT; in that case, calling
        # pts_bbox_head_infra.loss(...) with gt_labels_3d_infra=None crashes.
        has_infra_gt = (
            self.training_mode
            and gt_bboxes_3d_infra is not None
            and gt_labels_3d_infra is not None
            and gt_attr_labels_infra is not None
            and len(gt_bboxes_3d_infra) > 0
            and len(gt_labels_3d_infra) > 0
        )

        if has_infra_gt:
            infra_outs = {
                'all_cls_scores': torch.stack(outs['infra_pred_class'], dim=0),
                'all_bbox_preds': torch.stack(outs['infra_pred_coords'], dim=0)
            }

            # infra_outs must already be converted to VEHICLE frame before this block
            old_head_pc_range = copy.deepcopy(self.pts_bbox_head_infra.pc_range)
            old_bbox_coder_pc_range = copy.deepcopy(self.pts_bbox_head_infra.bbox_coder.pc_range)

            old_assigner_pc_range = None
            if hasattr(self.pts_bbox_head_infra, 'assigner') and hasattr(self.pts_bbox_head_infra.assigner, 'pc_range'):
                old_assigner_pc_range = copy.deepcopy(self.pts_bbox_head_infra.assigner.pc_range)

            veh_pc_range = outs["infra_veh_pc_range"]
            self.pts_bbox_head_infra.pc_range = copy.deepcopy(veh_pc_range)
            self.pts_bbox_head_infra.bbox_coder.pc_range = copy.deepcopy(veh_pc_range)
            if old_assigner_pc_range is not None:
                self.pts_bbox_head_infra.assigner.pc_range = copy.deepcopy(veh_pc_range)

            try:
                infra_loss_inputs = [
                    gt_bboxes_3d_infra, gt_labels_3d_infra, map_gt_bboxes_3d, map_gt_labels_3d,
                    infra_outs, ego_fut_trajs, ego_fut_masks, ego_fut_cmd, gt_attr_labels_infra
                ]

                losses_infra = self.pts_bbox_head_infra.loss(
                    *infra_loss_inputs, img_metas=infra_img_metas
                )
                # Infra bbox loss is ~14x larger than vehicle bbox loss because
                # early predictions fall outside infra_veh_pc_range, yielding
                # L1 errors > 1.0 per dimension.  Without down-weighting, infra
                # losses dominate ~86% of the total loss and destabilise the
                # vehicle head.  Scale to roughly match vehicle loss magnitude.
                losses.update({f"infra.{k}": v * self.infra_loss_weight for k, v in losses_infra.items()})
            finally:
                self.pts_bbox_head_infra.pc_range = old_head_pc_range
                self.pts_bbox_head_infra.bbox_coder.pc_range = old_bbox_coder_pc_range
                if old_assigner_pc_range is not None:
                    self.pts_bbox_head_infra.assigner.pc_range = old_assigner_pc_range
        
        if self.use_wandb and wandb is not None:
            # flatten nested loss dicts if any
            flat_losses = {}
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    flat_losses[k] = v.item()
                elif isinstance(v, dict):
                    for subk, subv in v.items():
                        flat_losses[f"{k}/{subk}"] = (
                            subv.item() if torch.is_tensor(subv) else subv
                        )
            wandb.log(flat_losses)

        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            veh_prev_bev = None # initialize with none
            infra_prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)

            veh_imgs_queue = imgs_queue[:,:1,...] #imgs_queue[:,:6,...] V2X-Sim
            infra_imgs_queue = imgs_queue[:,1:,...] #imgs_queue[:,6:,...]
            
            veh_img_feats_list = self.extract_feat(img=veh_imgs_queue, len_queue=len_queue, veh = True) # vehicle features
            
            for i in range(len_queue):
                veh_img_metas = [each[i] for each in img_metas_list]
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                veh_img_feats = [each_scale[:, i] for each_scale in veh_img_feats_list]
                veh_prev_bev = self.pts_bbox_head(
                        veh_img_feats, veh_img_metas, veh_prev_bev, only_bev=True)

            if self.training_mode: # V2X need to get infra bev for getting queries form infra
                infra_img_feats_list = self.extract_feat(img=infra_imgs_queue, len_queue=len_queue, veh = False) # infra features

                for i in range(len_queue):
                    #infra_img_metas = [each[i] for each in img_metas_list]
                    
                    
                    infra_img_metas = copy.deepcopy([each[i] for each in img_metas_list])
                    for b in range(len(infra_img_metas)):
                        infra_img_metas[b]['filename'] = [img_metas_list[b][i]['filename'][1]]
                        infra_img_metas[b]['ori_shape'] = [img_metas_list[b][i]['ori_shape'][1]]
                        infra_img_metas[b]['img_shape'] = [img_metas_list[b][i]['img_shape'][1]]
                        infra_img_metas[b]['lidar2img'] = [img_metas_list[b][i]['lidar2img'][1]]
                        infra_img_metas[b]['pts_filename'] = img_metas_list[b][i]['filename'][1].replace('/image/', '/velodyne/').rsplit('.', 1)[0] + '.pcd'
                        infra_img_metas[b]['can_bus'] = np.array([ 0.        ,  0.        ,  0.        , -0.2616553 , -0.2616553 ,
                                                                   -0.2616553 , -0.2616553 ,  0.        ,  0.        ,  0.        ,
                                                                   0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                                                                   0.        ,  3.67106654,  0.        ])
                    infra_img_feats = [each_scale[:, i] for each_scale in infra_img_feats_list]
                    
                    infra_prev_bev = self.pts_bbox_head_infra(
                            infra_img_feats, infra_img_metas, infra_prev_bev, only_bev=True)


            self.train()
            return veh_prev_bev, infra_prev_bev

    # @auto_fp16(apply_to=('img', 'points'))
    @force_fp32(apply_to=('img','points','prev_bev'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      map_gt_bboxes_3d=None,
                      map_gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      map_gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ego_his_trajs=None,
                      ego_fut_trajs=None,
                      ego_fut_masks=None,
                      ego_fut_cmd=None,
                      ego_lcf_feat=None,
                      gt_attr_labels=None, #ANIRUDH EDIT THE BELOW
                      gt_bboxes_3d_infra=None,
                      gt_labels_3d_infra=None,
                      gt_attr_labels_infra=None
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...] # size: B, len_queue, num_cams, channels, height, width
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        # prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        # import pdb;pdb.set_trace()
        #veh_prev_bev, infra_prev_bev = self.obtain_history_bev(prev_img, prev_img_metas) if len_queue > 1 else None
        if len_queue > 1:
            veh_prev_bev, infra_prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        else:
            veh_prev_bev, infra_prev_bev = None, None

        img_metas = [each[len_queue-1] for each in img_metas]

        veh_imgs = img[:, :1, ...] # change according to dataset maybe img[:, :6, ...] for V2X-Sim
        infra_imgs = img[:, 1:, ...] # change according to dataset maybe img[:, 6:, ...] for V2X-Sim
        
        # reverse the img to train the infra pipeline
        #veh_imgs = img # change according to dataset maybe img[:, :6, ...] for V2X-Sim
        #infra_imgs = img # change according to dataset maybe img[:, 6:, ...] for V2X-Sim
        #import pdb; pdb.set_trace()
        
        
        if self.training_mode: #vehicle and infrastructure included
            # fix vehicle pipeline for feature extraction: make only infra trainable
            infra_img_metas = copy.deepcopy(img_metas)
            infra_img_metas[0]['filename'] = [img_metas[0]['filename'][1]]
            infra_img_metas[0]['ori_shape'] = [img_metas[0]['ori_shape'][1]]
            infra_img_metas[0]['img_shape'] = [img_metas[0]['img_shape'][1]]
            infra_img_metas[0]['lidar2img'] = [img_metas[0]['lidar2img'][1]]
            infra_img_metas[0]['pts_filename'] = img_metas[0]['filename'][1].replace('/image/', '/velodyne/').rsplit('.', 1)[0] + '.pcd'
            infra_img_metas[0]['can_bus'] = np.array([ 0.        ,  0.        ,  0.        , -0.2616553 , -0.2616553 ,
       -0.2616553 , -0.2616553 ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  3.67106654,  0.        ])
            #with torch.no_grad():
            veh_img_feats_list = self.extract_feat(img=veh_imgs, img_metas=img_metas, veh = True)
            
            infra_img_feats_list = self.extract_feat(img=infra_imgs, img_metas=infra_img_metas, veh = False)    
                
            img_feats = [veh_img_feats_list,infra_img_feats_list]

        else: #only vehicle
            img_feats = self.extract_feat(img=veh_imgs, img_metas=img_metas, veh = True) 

        # img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d, gt_labels_3d,
                                            map_gt_bboxes_3d, map_gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, map_gt_bboxes_ignore, veh_prev_bev,infra_prev_bev,
                                            ego_his_trajs=ego_his_trajs, ego_fut_trajs=ego_fut_trajs,
                                            ego_fut_masks=ego_fut_masks, ego_fut_cmd=ego_fut_cmd,
                                            ego_lcf_feat=ego_lcf_feat, gt_attr_labels=gt_attr_labels,
                                            gt_bboxes_3d_infra=gt_bboxes_3d_infra,gt_labels_3d_infra=gt_labels_3d_infra,gt_attr_labels_infra=gt_attr_labels_infra)

        losses.update(losses_pts)
        return losses

    def forward_test(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        img=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        **kwargs
    ):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
            self.prev_frame_info['prev_infra_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None
            self.prev_frame_info['prev_infra_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        new_prev_bev, new_prev_infra_bev, bbox_results = self.simple_test(
            img_metas=img_metas[0],
            img=img[0],
            prev_bev=self.prev_frame_info['prev_bev'],
            prev_infra_bev = self.prev_frame_info['prev_infra_bev'],
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            ego_his_trajs=ego_his_trajs[0],
            ego_fut_trajs=ego_fut_trajs[0],
            ego_fut_cmd=ego_fut_cmd[0],
            ego_lcf_feat=ego_lcf_feat[0],
            gt_attr_labels=gt_attr_labels,
            **kwargs
        )
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        self.prev_frame_info['prev_infra_bev'] = new_prev_infra_bev

        return bbox_results

    def simple_test(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        img=None,
        prev_bev=None,
        prev_infra_bev=None,
        points=None,
        fut_valid_flag=None,
        rescale=False,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        **kwargs
    ):
        """Test function without augmentaiton."""
        veh_imgs = img[:, :1, ...] # change according to dataset maybe img[:, :6, ...] for V2X-Sim
        infra_imgs = img[:, 1:, ...] # change according to dataset maybe img[:, 6:, ...] for V2X-Sim

        
        if self.training_mode: #vehicle and infrastructure included
            # fix vehicle pipeline for feature extraction: make only infra trainable
            #with torch.no_grad():
        
            infra_img_metas = copy.deepcopy(img_metas)
            infra_img_metas[0]['filename'] = [img_metas[0]['filename'][1]]
            infra_img_metas[0]['ori_shape'] = [img_metas[0]['ori_shape'][1]]
            infra_img_metas[0]['img_shape'] = [img_metas[0]['img_shape'][1]]
            infra_img_metas[0]['lidar2img'] = [img_metas[0]['lidar2img'][1]]
            infra_img_metas[0]['pts_filename'] = img_metas[0]['filename'][1].replace('/image/', '/velodyne/').rsplit('.', 1)[0] + '.pcd'
            infra_img_metas[0]['can_bus'] = np.array([ 0.        ,  0.        ,  0.        , -0.2616553 , -0.2616553 ,
       -0.2616553 , -0.2616553 ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  3.67106654,  0.        ])
            
            veh_img_feats_list = self.extract_feat(img=veh_imgs, img_metas=img_metas, veh = True)
            
            infra_img_feats_list = self.extract_feat(img=infra_imgs, img_metas=infra_img_metas, veh = False)    
                
            img_feats = [veh_img_feats_list,infra_img_feats_list]

        else: #only vehicle
            
            img_feats = self.extract_feat(img=veh_imgs, img_metas=img_metas, veh = True)
            
        #img_feats = self.extract_feat(img=img, img_metas=img_metas, veh=True)
        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, new_prev_infra_bev, bbox_pts, metric_dict = self.simple_test_pts(
            img_feats,
            img_metas,
            gt_bboxes_3d,
            gt_labels_3d,
            prev_bev,
            prev_infra_bev=prev_infra_bev,
            fut_valid_flag=fut_valid_flag,
            rescale=rescale,
            start=None,
            ego_his_trajs=ego_his_trajs,
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_cmd=ego_fut_cmd,
            ego_lcf_feat=ego_lcf_feat,
            gt_attr_labels=gt_attr_labels,
        )
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['metric_results'] = metric_dict
        
        return new_prev_bev, new_prev_infra_bev, bbox_list

    def plot_3d_centers(self, gt_centers, pred_centers, color='red', s_gt=40, s_pred=60):
        """
        Plot 3D GT centers as black spheres and predicted centers as colored triangles.

        Args:
            gt_centers (np.ndarray or tensor): shape (N, 3)
            pred_centers (np.ndarray or tensor): shape (M, 3)
            color (str or tuple): color for predicted centers
            s_gt (int): marker size for GT spheres
            s_pred (int): marker size for predicted triangles
        """

        # Convert torch tensors to numpy if needed
        if hasattr(gt_centers, "detach"):
            gt_centers = gt_centers.detach().cpu().numpy()
        if hasattr(pred_centers, "detach"):
            pred_centers = pred_centers.detach().cpu().numpy()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # --- GT: black spheres ---
        ax.scatter(
            gt_centers[:, 0], gt_centers[:, 1], gt_centers[:, 2],
            c='black', s=s_gt, marker='o', label='GT Centers'
        )

        # --- Pred: colored triangles ---
        ax.scatter(
            pred_centers[:, 0], pred_centers[:, 1], pred_centers[:, 2],
            c=color, s=s_pred, marker='^', label='Pred Centers'
        )

        # Labels and aesthetics
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_3d_bboxes(self, gt_bboxes, pred_bboxes, color='red', gt_color='black'):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        def manual_boxes_to_corners(boxes):
            if hasattr(boxes, "tensor"):
                boxes = boxes.tensor
            if isinstance(boxes, np.ndarray):
                boxes = torch.from_numpy(boxes)
            boxes = boxes.detach().cpu().float()

            if boxes.numel() == 0:
                return np.zeros((0, 8, 3), dtype=np.float32)

            boxes = boxes[:, :7]

            centers = boxes[:, 0:3]
            dims = boxes[:, 3:6]
            yaws = boxes[:, 6]

            dx, dy, dz = dims[:, 0], dims[:, 1], dims[:, 2]

            x_corners = torch.stack([
                dx/2,  dx/2, -dx/2, -dx/2,
                dx/2,  dx/2, -dx/2, -dx/2
            ], dim=1)

            y_corners = torch.stack([
                dy/2, -dy/2, -dy/2,  dy/2,
                dy/2, -dy/2, -dy/2,  dy/2
            ], dim=1)

            z_corners = torch.stack([
                -dz/2, -dz/2, -dz/2, -dz/2,
                dz/2,  dz/2,  dz/2,  dz/2
            ], dim=1)

            corners = torch.stack([x_corners, y_corners, z_corners], dim=-1)

            cos_yaw = torch.cos(yaws)
            sin_yaw = torch.sin(yaws)

            rot = torch.zeros((boxes.shape[0], 3, 3), dtype=torch.float32)
            rot[:, 0, 0] = cos_yaw
            rot[:, 0, 1] = -sin_yaw
            rot[:, 1, 0] = sin_yaw
            rot[:, 1, 1] = cos_yaw
            rot[:, 2, 2] = 1.0

            corners = torch.matmul(corners, rot.transpose(1, 2))
            corners = corners + centers[:, None, :]
            return corners.numpy()

        def to_corners(boxes, use_native_corners=False):
            if hasattr(boxes, "tensor"):
                if boxes.tensor.numel() == 0:
                    return np.zeros((0, 8, 3), dtype=np.float32)

            if use_native_corners:
                corners = boxes.corners
                if hasattr(corners, "detach"):
                    corners = corners.detach().cpu().numpy()
                return corners

            return manual_boxes_to_corners(boxes)

        # GT can be empty, so safer to use manual conversion
        gt_corners = to_corners(gt_bboxes, use_native_corners=False)

        # Pred boxes seem to support .corners in your case
        pred_corners = to_corners(pred_bboxes, use_native_corners=True)

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        def draw_boxes(corners_set, edge_color, label):
            if corners_set is None or len(corners_set) == 0:
                return

            first = True
            for box in corners_set:
                for i, j in edges:
                    ax.plot(
                        [box[i, 0], box[j, 0]],
                        [box[i, 1], box[j, 1]],
                        [box[i, 2], box[j, 2]],
                        c=edge_color,
                        linewidth=1.5,
                        label=label if first and (i, j) == (0, 1) else None
                    )
                first = False

        draw_boxes(gt_corners, gt_color, 'GT Boxes')
        draw_boxes(pred_corners, color, 'Pred Boxes')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()    


    def simple_test_pts(
        self,
        x,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        prev_bev=None,
        prev_infra_bev=None,
        fut_valid_flag=None,
        rescale=False,
        start=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
    ):
        """Test function"""
        mapped_class_names = [
            'car', 'truck', 'construction_vehicle', 'bus',
            'trailer', 'barrier', 'motorcycle', 'bicycle', 
            'pedestrian', 'traffic_cone'
        ]
        if not self.training_mode:
            outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev, ego_his_trajs=ego_his_trajs, ego_lcf_feat=ego_lcf_feat)
            new_prev_infra_bev = None
        else:
            veh_feats = x[0]
            infra_feats = x[1]

            
            # getting queries from infrastructure using infra head
            
            infra_prev_bev = prev_infra_bev
            infra_img_metas = copy.deepcopy(img_metas)
            infra_img_metas[0]['filename'] = [img_metas[0]['filename'][1]]
            infra_img_metas[0]['ori_shape'] = [img_metas[0]['ori_shape'][1]]
            infra_img_metas[0]['img_shape'] = [img_metas[0]['img_shape'][1]]
            infra_img_metas[0]['lidar2img'] = [img_metas[0]['lidar2img'][1]]
            infra_img_metas[0]['pts_filename'] = img_metas[0]['filename'][1].replace('/image/', '/velodyne/').rsplit('.', 1)[0] + '.pcd'
            infra_img_metas[0]['can_bus'] = np.array([ 0.        ,  0.        ,  0.        , -0.2616553 , -0.2616553 ,
       -0.2616553 , -0.2616553 ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  3.67106654,  0.        ])
            #import pdb; pdb.set_trace()
            #infra_queries = self.pts_bbox_head_infra(infra_feats, infra_img_metas, infra_prev_bev,
                                #ego_his_trajs=ego_his_trajs, ego_lcf_feat=ego_lcf_feat, training_mode = self.training_mode)

            infra_outs = self.pts_bbox_head_infra(infra_feats, infra_img_metas, infra_prev_bev,
                                ego_his_trajs=ego_his_trajs, ego_lcf_feat=ego_lcf_feat, training_mode = self.training_mode)

            infra_queries = infra_outs['queries']
            infra_preds = infra_outs['preds_dicts']

            outs = self.pts_bbox_head(veh_feats, img_metas, prev_bev, ego_his_trajs=ego_his_trajs, ego_lcf_feat=ego_lcf_feat, infra_queries = infra_queries, training_mode = self.training_mode, gt_bboxes_3d=gt_bboxes_3d)
            # import pdb; pdb.set_trace()
            new_prev_infra_bev = self.pts_bbox_head_infra(infra_feats, infra_img_metas, prev_bev=prev_infra_bev, only_bev=True)
        
        infra_preds['all_traj_preds'] = outs['all_traj_preds']
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        infra_bbox_list = self.pts_bbox_head_infra.get_bboxes(infra_preds, infra_img_metas, rescale=rescale)

        # infra_last = outs['infra_pred_coords'][-1] # last layer
        # infra_coords = torch.cat([infra_last[..., 0:2], infra_last[..., 4:5]], dim=-1)

        # veh_last = outs['all_bbox_preds'][-1]
        # veh_coords = torch.cat([veh_last[..., 0:2], veh_last[..., 4:5]], dim=-1)
        # gt_centers = gt_bboxes_3d[0][0].center

        # self.plot_3d_centers(gt_centers, infra_coords,color='blue')
        # self.plot_3d_centers(gt_centers, veh_coords,color='red')

        # plotting boxes
        # gt_boxes = gt_bboxes_3d[0][0]
        # veh_pred_boxes = bbox_list[0][0]
        # veh_mask = bbox_list[0][1]>0.6
        # veh_pred_boxes = veh_pred_boxes[veh_mask]
        # import pdb; pdb.set_trace()
        # infra_pred_boxes = infra_bbox_list[0][0]
        
        bbox_results = []
        for i, (bboxes, scores, labels, trajs, map_bboxes, \
                map_scores, map_labels, map_pts) in enumerate(bbox_list):
            bbox_result = bbox3d2result(bboxes, scores, labels)
            bbox_result['trajs_3d'] = trajs.cpu()
            map_bbox_result = self.map_pred2result(map_bboxes, map_scores, map_labels, map_pts)
            bbox_result.update(map_bbox_result)
            bbox_result['ego_fut_preds'] = outs['ego_fut_preds'][i].cpu()
            bbox_result['ego_fut_cmd'] = ego_fut_cmd.cpu()
            bbox_results.append(bbox_result)

        
        assert len(bbox_results) == 1, 'only support batch_size=1 now'
        score_threshold = 0.6

        # filtering out lower confidence detections like UniV2X
        # for i in range(len(bbox_results)):
        #     bbox_result = bbox_results[i]
        #     # import pdb; pdb.set_trace()
        #     mask = bbox_result['scores_3d']>0.01
        #     bbox_result['boxes_3d'] = bbox_result['boxes_3d'][mask]
        #     bbox_result['scores_3d'] = bbox_result['scores_3d'][mask]
        #     bbox_result['labels_3d'] = bbox_result['labels_3d'][mask]
        #     bbox_result['trajs_3d'] = bbox_result['trajs_3d'][mask]
        #     bbox_results[i] = bbox_result
        
        with torch.no_grad():
            c_bbox_results = copy.deepcopy(bbox_results)

            bbox_result = c_bbox_results[0]
            gt_bbox = gt_bboxes_3d[0][0]
            gt_label = gt_labels_3d[0][0].to('cpu')
            gt_attr_label = gt_attr_labels[0][0].to('cpu')
            fut_valid_flag = bool(fut_valid_flag[0][0])
            # filter pred bbox by score_threshold
            # import pdb; pdb.set_trace()
            mask = bbox_result['scores_3d'] > score_threshold
            bbox_result['boxes_3d'] = bbox_result['boxes_3d'][mask]
            bbox_result['scores_3d'] = bbox_result['scores_3d'][mask]
            bbox_result['labels_3d'] = bbox_result['labels_3d'][mask]
            bbox_result['trajs_3d'] = bbox_result['trajs_3d'][mask]

            matched_bbox_result = self.assign_pred_to_gt_vip3d(
                bbox_result, gt_bbox, gt_label)

            metric_dict = self.compute_motion_metric_vip3d(
                gt_bbox, gt_label, gt_attr_label, bbox_result,
                matched_bbox_result, mapped_class_names)

            # ego planning metric
            assert ego_fut_trajs.shape[0] == 1, 'only support batch_size=1 for testing'
            ego_fut_preds = bbox_result['ego_fut_preds']
            ego_fut_trajs = ego_fut_trajs[0, 0]
            ego_fut_cmd = ego_fut_cmd[0, 0, 0]
            ego_fut_cmd_idx = torch.nonzero(ego_fut_cmd)[0, 0]
            ego_fut_pred = ego_fut_preds[ego_fut_cmd_idx]
            ego_fut_pred = ego_fut_pred.cumsum(dim=-2)
            ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)

            metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
                pred_ego_fut_trajs = ego_fut_pred[None],
                gt_ego_fut_trajs = ego_fut_trajs[None],
                gt_agent_boxes = gt_bbox,
                gt_agent_feats = gt_attr_label.unsqueeze(0),
                fut_valid_flag = fut_valid_flag
            )
            metric_dict.update(metric_dict_planner_stp3)
        
        return outs['bev_embed'],new_prev_infra_bev, bbox_results, metric_dict

    def map_pred2result(self, bboxes, scores, labels, pts, attrs=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
            labels (torch.Tensor): Labels with shape of (n, ).
            scores (torch.Tensor): Scores with shape of (n, ).
            attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (torch.Tensor): 3D boxes.
                - scores (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """
        result_dict = dict(
            map_boxes_3d=bboxes.to('cpu'),
            map_scores_3d=scores.cpu(),
            map_labels_3d=labels.cpu(),
            map_pts_3d=pts.to('cpu'))

        if attrs is not None:
            result_dict['map_attrs_3d'] = attrs.cpu()

        return result_dict

    def assign_pred_to_gt_vip3d(
        self,
        bbox_result,
        gt_bbox,
        gt_label,
        match_dis_thresh=2.0
    ):
        """Assign pred boxs to gt boxs according to object center preds in lcf.
        Args:
            bbox_result (dict): Predictions.
                'boxes_3d': (LiDARInstance3DBoxes)
                'scores_3d': (Tensor), [num_pred_bbox]
                'labels_3d': (Tensor), [num_pred_bbox]
                'trajs_3d': (Tensor), [fut_ts*2]
            gt_bboxs (LiDARInstance3DBoxes): GT Bboxs.
            gt_label (Tensor): GT labels for gt_bbox, [num_gt_bbox].
            match_dis_thresh (float): dis thresh for determine a positive sample for a gt bbox.

        Returns:
            matched_bbox_result (np.array): assigned pred index for each gt box [num_gt_bbox].
        """     
        dynamic_list = [0,1,3,4,6,7,8]
        matched_bbox_result = torch.ones(
            (len(gt_bbox)), dtype=torch.long) * -1  # -1: not assigned
        gt_centers = gt_bbox.center[:, :2]
        pred_centers = bbox_result['boxes_3d'].center[:, :2]
        dist = torch.linalg.norm(pred_centers[:, None, :] - gt_centers[None, :, :], dim=-1)
        pred_not_dyn = [label not in dynamic_list for label in bbox_result['labels_3d']]
        gt_not_dyn = [label not in dynamic_list for label in gt_label]
        dist[pred_not_dyn] = 1e6
        dist[:, gt_not_dyn] = 1e6
        dist[dist > match_dis_thresh] = 1e6

        r_list, c_list = linear_sum_assignment(dist)

        for i in range(len(r_list)):
            if dist[r_list[i], c_list[i]] <= match_dis_thresh:
                matched_bbox_result[c_list[i]] = r_list[i]

        return matched_bbox_result

    def compute_motion_metric_vip3d(
        self,
        gt_bbox,
        gt_label,
        gt_attr_label,
        pred_bbox,
        matched_bbox_result,
        mapped_class_names,
        match_dis_thresh=2.0,
    ):
        """Compute EPA metric for one sample.
        Args:
            gt_bboxs (LiDARInstance3DBoxes): GT Bboxs.
            gt_label (Tensor): GT labels for gt_bbox, [num_gt_bbox].
            pred_bbox (dict): Predictions.
                'boxes_3d': (LiDARInstance3DBoxes)
                'scores_3d': (Tensor), [num_pred_bbox]
                'labels_3d': (Tensor), [num_pred_bbox]
                'trajs_3d': (Tensor), [fut_ts*2]
            matched_bbox_result (np.array): assigned pred index for each gt box [num_gt_bbox].
            match_dis_thresh (float): dis thresh for determine a positive sample for a gt bbox.

        Returns:
            EPA_dict (dict): EPA metric dict of each cared class.
        """
        motion_cls_names = ['car', 'pedestrian']
        motion_metric_names = ['gt', 'cnt_ade', 'cnt_fde', 'hit',
                               'fp', 'ADE', 'FDE', 'MR']
        
        metric_dict = {}
        for met in motion_metric_names:
            for cls in motion_cls_names:
                metric_dict[met+'_'+cls] = 0.0

        veh_list = [0,1,3,4]
        ignore_list = ['construction_vehicle', 'barrier',
                       'traffic_cone', 'motorcycle', 'bicycle']

        for i in range(pred_bbox['labels_3d'].shape[0]):
            pred_bbox['labels_3d'][i] = 0 if pred_bbox['labels_3d'][i] in veh_list else pred_bbox['labels_3d'][i]
            box_name = mapped_class_names[pred_bbox['labels_3d'][i]]
            if box_name in ignore_list:
                continue
            if i not in matched_bbox_result:
                metric_dict['fp_'+box_name] += 1

        for i in range(gt_label.shape[0]):
            gt_label[i] = 0 if gt_label[i] in veh_list else gt_label[i]
            box_name = mapped_class_names[gt_label[i]]
            if box_name in ignore_list:
                continue
            gt_fut_masks = gt_attr_label[i][self.fut_ts*2:self.fut_ts*3]
            num_valid_ts = sum(gt_fut_masks==1)
            if num_valid_ts == self.fut_ts:
                metric_dict['gt_'+box_name] += 1
            if matched_bbox_result[i] >= 0 and num_valid_ts > 0:
                metric_dict['cnt_ade_'+box_name] += 1
                m_pred_idx = matched_bbox_result[i]
                gt_fut_trajs = gt_attr_label[i][:self.fut_ts*2].reshape(-1, 2)
                gt_fut_trajs = gt_fut_trajs[:num_valid_ts]
                pred_fut_trajs = pred_bbox['trajs_3d'][m_pred_idx].reshape(self.fut_mode, self.fut_ts, 2)
                pred_fut_trajs = pred_fut_trajs[:, :num_valid_ts, :]
                gt_fut_trajs = gt_fut_trajs.cumsum(dim=-2)
                pred_fut_trajs = pred_fut_trajs.cumsum(dim=-2)
                gt_fut_trajs = gt_fut_trajs + gt_bbox[i].center[0, :2]
                pred_fut_trajs = pred_fut_trajs + pred_bbox['boxes_3d'][int(m_pred_idx)].center[0, :2]

                dist = torch.linalg.norm(gt_fut_trajs[None, :, :] - pred_fut_trajs, dim=-1)
                ade = dist.sum(-1) / num_valid_ts
                ade = ade.min()

                metric_dict['ADE_'+box_name] += ade
                if num_valid_ts == self.fut_ts:
                    fde = dist[:, -1].min()
                    metric_dict['cnt_fde_'+box_name] += 1
                    metric_dict['FDE_'+box_name] += fde
                    if fde <= match_dis_thresh:
                        metric_dict['hit_'+box_name] += 1
                    else:
                        metric_dict['MR_'+box_name] += 1

        return metric_dict

    ### same planning metric as stp3
    def compute_planner_metric_stp3(
        self,
        pred_ego_fut_trajs,
        gt_ego_fut_trajs,
        gt_agent_boxes,
        gt_agent_feats,
        fut_valid_flag
    ):
        """Compute planner metric for one sample same as stp3."""
        metric_dict = {
            'plan_L2_1s':0,
            'plan_L2_2s':0,
            'plan_L2_3s':0,
            'plan_obj_col_1s':0,
            'plan_obj_col_2s':0,
            'plan_obj_col_3s':0,
            'plan_obj_box_col_1s':0,
            'plan_obj_box_col_2s':0,
            'plan_obj_box_col_3s':0,
        }
        metric_dict['fut_valid_flag'] = fut_valid_flag
        future_second = 3
        assert pred_ego_fut_trajs.shape[0] == 1, 'only support bs=1'
        if self.planning_metric is None:
            self.planning_metric = PlanningMetric()
        segmentation, pedestrian = self.planning_metric.get_label(
            gt_agent_boxes, gt_agent_feats)
        occupancy = torch.logical_or(segmentation, pedestrian)

        for i in range(future_second):
            if fut_valid_flag:
                cur_time = (i+1)*2
                traj_L2 = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time]
                )
                obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach(),
                    gt_ego_fut_trajs[:, :cur_time],
                    occupancy)
                metric_dict['plan_L2_{}s'.format(i+1)] = traj_L2
                metric_dict['plan_obj_col_{}s'.format(i+1)] = obj_coll.mean().item()
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = obj_box_coll.mean().item()
            else:
                metric_dict['plan_L2_{}s'.format(i+1)] = 0.0
                metric_dict['plan_obj_col_{}s'.format(i+1)] = 0.0
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = 0.0
            
        return metric_dict

    def set_epoch(self, epoch): 
        self.pts_bbox_head.epoch = epoch
