# All modification made by Kneron Corp.: Copyright (c) 2022 Kneron Corp.
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import shutil
import warnings

import mmcv
import torch
from mmcv.runner import get_dist_info
from mmcv.utils import DictAction

from mmseg.apis import single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models.segmentors.base import ONNXRuntimeSegmentorKN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='mmseg backend test (and eval)')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('model', help='Input model file (onnx only)')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
        'not be supported in version v0.22.0. Override some settings in the '
        'used config, the key-value pair in xxx=yyy format will be merged '
        'into config file. If the value to be overwritten is a list, it '
        'should be like key="[a,b]" or key=a,b It also allows nested '
        'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
        'marks are necessary and that no white space is allowed.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=None,
        help='input image height and width.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options. '
                      '--options will not be supported in version v0.22.0.')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    if args.shape is not None:

        if len(args.shape) == 1:
            shape = (args.shape[0], args.shape[0])
        elif len(args.shape) == 2:
            shape = (args.shape[1], args.shape[0])
        else:
            raise ValueError('invalid input shape')

        test_mode = cfg.model.test_cfg.mode
        if test_mode == 'slide':
            warnings.warn(
                "We suggest you NOT assigning shape when exporting "
                "slide-mode models. Assigning shape to slide-mode models "
                "may result in unexpected results. To see which mode the "
                "model is using, check cfg.model.test_cfg.mode, which "
                "should be either 'whole' or 'slide'."
            )
            cfg.model.test_cfg['crop_size'] = shape
        else:
            cfg.test_pipeline[1]['img_scale'] = shape
        cfg.data.test['pipeline'][1]['img_scale'] = shape

    # init distributed env first, since logger depends on the dist info.
    distributed = False

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # load onnx config and meta
    cfg.model.train_cfg = None

    model = ONNXRuntimeSegmentorKN(args.model, cfg=cfg, device_id=0)

    model.CLASSES = dataset.CLASSES
    model.PALETTE = dataset.PALETTE

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()
    eval_kwargs = {} if args.eval_options is None else args.eval_options

    # Deprecated
    efficient_test = eval_kwargs.get('efficient_test', False)
    if efficient_test:
        warnings.warn(
            '"efficient_test=True" does not have effect in '
            'tools/test_kneron.py, the evaluation and format '
            'results are CPU memory efficient by default')

    eval_on_format_results = (
        args.eval is not None and 'cityscapes' in args.eval)
    if eval_on_format_results:
        assert len(args.eval) == 1, 'eval on format results is not ' \
                                    'applicable for metrics other than ' \
                                    'cityscapes'
    if args.format_only or eval_on_format_results:
        if 'imgfile_prefix' in eval_kwargs:
            tmpdir = eval_kwargs['imgfile_prefix']
        else:
            tmpdir = '.format_cityscapes'
            eval_kwargs.setdefault('imgfile_prefix', tmpdir)
        mmcv.mkdir_or_exist(tmpdir)
    else:
        tmpdir = None

    results = single_gpu_test(
        model,
        data_loader,
        args.show,
        args.show_dir,
        False,
        args.opacity,
        pre_eval=args.eval is not None and not eval_on_format_results,
        format_only=args.format_only or eval_on_format_results,
        format_args=eval_kwargs)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            warnings.warn(
                'The behavior of ``args.out`` has been changed since MMSeg '
                'v0.16, the pickled outputs could be seg map as type of '
                'np.array, pre-eval results or file paths for '
                '``dataset.format_results()``.')
            print(f'\nwriting results to {args.out}')
            mmcv.dump(results, args.out)
        if args.eval:
            dataset.evaluate(results, args.eval, **eval_kwargs)
        if tmpdir is not None and eval_on_format_results:
            # remove tmp dir when cityscapes evaluation
            shutil.rmtree(tmpdir)


if __name__ == '__main__':
    main()
