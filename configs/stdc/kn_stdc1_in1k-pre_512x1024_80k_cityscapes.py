checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/stdc/stdc1_20220308-5368626c.pth'  # noqa
_base_ = [
    '../_base_/models/stdc.py', '../_base_/datasets/kn_cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
lr_config = dict(warmup='linear', warmup_iters=1000)
data = dict(
    samples_per_gpu=12,
    workers_per_gpu=4,
)
model = dict(
    backbone=dict(
        backbone_cfg=dict(
            init_cfg=dict(type='Pretrained', checkpoint=checkpoint))))
