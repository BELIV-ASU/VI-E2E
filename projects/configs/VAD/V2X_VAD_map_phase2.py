"""Phase-2 map training config.

Load ep200 weights with --load-from, then run 100 fresh epochs with:
  - 4× higher effective LR for map decoder (2e-4 vs 5e-5 base)
  - curvature smoothness loss enabled (weight 0.1, up from 0.05 in base)
  - fresh CosineAnnealing schedule so LR doesn't start at near-zero

Usage:
    python tools/train.py projects/configs/VAD/V2X_VAD_map_phase2.py \
        --load-from /home/jingxiong/v2xvad_plan_test/epoch_200.pth \
        --work-dir /home/jingxiong/v2xvad_map_phase2
"""

_base_ = ['V2X_VAD_base_e2e_patch.py']

total_epochs = 100

# Fresh cosine schedule.  min_lr_ratio=5e-2 (not 1e-3) so LR doesn't
# collapse too far in just 100 epochs.
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=300,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=5e-2)

# Lower base LR protects planning/traj params (Adam state resets with
# load_from); map decoder gets 4× boost → 4*5e-5 = 2e-4 effective LR,
# matching the original training peak and giving the undertrained layers
# a proper fresh start.
optimizer = dict(
    type='AdamW',
    lr=5e-5,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            # ── Frozen: feature extraction ──────────────────────────────────
            'img_backbone': dict(lr_mult=0.0, decay_mult=0.0),
            'img_backbone_infra': dict(lr_mult=0.0, decay_mult=0.0),
            'img_neck': dict(lr_mult=0.0, decay_mult=0.0),
            'img_neck_infra': dict(lr_mult=0.0, decay_mult=0.0),
            'pts_bbox_head.transformer.encoder': dict(lr_mult=0.0, decay_mult=0.0),
            'pts_bbox_head.transformer.decoder': dict(lr_mult=0.0, decay_mult=0.0),
            'pts_bbox_head.transformer.level_embeds': dict(lr_mult=0.0, decay_mult=0.0),
            'pts_bbox_head.transformer.can_bus_mlp': dict(lr_mult=0.0, decay_mult=0.0),
            'pts_bbox_head.bev_embedding': dict(lr_mult=0.0, decay_mult=0.0),
            'pts_bbox_head.query_embedding': dict(lr_mult=0.0, decay_mult=0.0),
            'pts_bbox_head.cls_branches': dict(lr_mult=0.0, decay_mult=0.0),
            'pts_bbox_head.reg_branches': dict(lr_mult=0.0, decay_mult=0.0),
            'pts_bbox_head_infra': dict(lr_mult=0.0, decay_mult=0.0),
            # ── Frozen: V2X fusion ───────────────────────────────────────────
            'pts_bbox_head.agent_fusion_decoder': dict(lr_mult=0.0, decay_mult=0.0),
            'pts_bbox_head.cross_agent_fusion': dict(lr_mult=0.0, decay_mult=0.0),
            'pts_bbox_head.vi_map_fuser': dict(lr_mult=0.0, decay_mult=0.0),
            'pts_bbox_head.bev_embed_linear': dict(lr_mult=0.0, decay_mult=0.0),
            'pts_bbox_head.bev_pos_linear': dict(lr_mult=0.0, decay_mult=0.0),
            'pts_bbox_head.cls_branches_fuse': dict(lr_mult=0.0, decay_mult=0.0),
            'pts_bbox_head.reg_branches_fuse': dict(lr_mult=0.0, decay_mult=0.0),
            'pts_bbox_head.agent_fus_mlp': dict(lr_mult=0.0, decay_mult=0.0),
            # ── Map decoder: 4× boosted ──────────────────────────────────────
            # layers 0-1 have requires_grad=False (VAD.py freeze), so the
            # 4× mult only affects layers 2-5 that are actually trainable.
            'pts_bbox_head.transformer.map_decoder': dict(lr_mult=4.0),
            'pts_bbox_head.map_instance_embedding': dict(lr_mult=4.0),
            'pts_bbox_head.map_pts_embedding': dict(lr_mult=4.0),
        }
    )
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

# Validate every 10 epochs — phase2 is short so more frequent evals
# are worth the RAM cost (Firefox closed, workers_per_gpu=2 still in base).
evaluation = dict(interval=10, metric='bbox', map_metric='chamfer')

# Increase curvature weight for phase2 — this is the focused map phase.
model = dict(
    pts_bbox_head=dict(
        loss_map_curvature=dict(type='PtsCurvatureLoss', loss_weight=0.1)
    )
)
