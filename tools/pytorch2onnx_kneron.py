# Copyright (c) OpenMMLab. All rights reserved.
# Original: tools/pytorch2onnx.py, modified by Kneron
import argparse

import warnings
import os
import onnx
import mmcv
import numpy as np
import onnxruntime as rt
import torch
import torch._C
import torch.serialization
from mmcv import DictAction
from mmcv.onnx import register_extra_symbolics
from mmcv.runner import load_checkpoint
from torch import nn

from mmseg.apis import show_result_pyplot
from mmseg.apis.inference import LoadImage
from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor

from optimizer_scripts.tools import other
from optimizer_scripts.pytorch_exported_onnx_preprocess import (
    torch_exported_onnx_flow,
)

torch.manual_seed(3)


def _parse_normalize_cfg(test_pipeline):
    transforms = None
    for pipeline in test_pipeline:
        if 'transforms' in pipeline:
            transforms = pipeline['transforms']
            break
    assert transforms is not None, 'Failed to find `transforms`'
    norm_config_li = [_ for _ in transforms if _['type'] == 'Normalize']
    assert len(norm_config_li) == 1, '`norm_config` should only have one'
    norm_config = norm_config_li[0]
    return norm_config


def _convert_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


def _demo_mm_inputs(input_shape):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
        num_classes (int):
            number of semantic classes
    """
    (N, C, H, W) = input_shape
    rng = np.random.RandomState(0)
    img = torch.FloatTensor(rng.rand(*input_shape))
    return img


def _prepare_input_img(img_path,
                       test_pipeline,
                       shape=None):
    # build the data pipeline
    if shape is not None:
        test_pipeline[1]['img_scale'] = (shape[1], shape[0])
    test_pipeline[1]['transforms'][0]['keep_ratio'] = False
    test_pipeline = [LoadImage()] + test_pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img_path)
    data = test_pipeline(data)
    img = torch.FloatTensor(data['img']).unsqueeze_(0)
    return img


def pytorch2onnx(model,
                 img,
                 norm_cfg=None,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False):
    """Export Pytorch model to ONNX model and verify the outputs are same
    between Pytorch and ONNX.

    Args:
        model (nn.Module): Pytorch model we want to export.
        img (dict): Input tensor (1xCxHxW)
        opset_version (int): The onnx op version. Default: 11.
        show (bool): Whether print the computation graph. Default: False.
        output_file (string): The path to where we store the output ONNX model.
            Default: `tmp.onnx`.
        verify (bool): Whether compare the outputs between Pytorch and ONNX.
            Default: False.
    """
    model.cpu().eval()

    if isinstance(model.decode_head, nn.ModuleList):
        num_classes = model.decode_head[-1].num_classes
    else:
        num_classes = model.decode_head.num_classes

    # replace original forward function
    model.forward = model.forward_dummy
    origin_forward = model.forward

    register_extra_symbolics(opset_version)
    with torch.no_grad():
        torch.onnx.export(
            model, img,
            output_file,
            input_names=['input'],
            output_names=['output'],
            export_params=True,
            keep_initializers_as_inputs=False,
            verbose=show,
            opset_version=opset_version,
            dynamic_axes=None)
        print(f'Successfully exported ONNX model: {output_file}')
    model.forward = origin_forward
    # NOTE: optimizing onnx for kneron inference
    m = onnx.load(output_file)
    # NOTE: PyTorch 1.10.x exports onnx ir_version == 7 for opset 11,
    #       but should be ir_version == 6
    if opset_version == 11:
        m.ir_version = 6
    m = torch_exported_onnx_flow(m, disable_fuse_bn=False)
    onnx.save(m, output_file)
    print(f'{output_file} optimized by KNERON successfully.')

    if verify:
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # check the numerical value
        # get pytorch output
        with torch.no_grad():
            pytorch_result = model(img).numpy()

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 1)
        sess = rt.InferenceSession(
            output_file, providers=['CPUExecutionProvider']
        )
        onnx_result = sess.run(
            None, {net_feed_input[0]: img.detach().numpy()})[0]
        # show segmentation results
        if show:
            import cv2
            img = img[0][:3, ...].permute(1, 2, 0) * 255
            img = img.detach().numpy().astype(np.uint8)
            ori_shape = img.shape[:2]

            # resize onnx_result to ori_shape
            onnx_result_ = onnx_result[0].argmax(0)
            onnx_result_ = cv2.resize(onnx_result_.astype(np.uint8),
                                      (ori_shape[1], ori_shape[0]))
            show_result_pyplot(
                model,
                img, (onnx_result_, ),
                palette=model.PALETTE,
                block=False,
                title='ONNXRuntime',
                opacity=0.5)

            # resize pytorch_result to ori_shape
            pytorch_result_ = pytorch_result.squeeze().argmax(0)
            pytorch_result_ = cv2.resize(pytorch_result_.astype(np.uint8),
                                         (ori_shape[1], ori_shape[0]))
            show_result_pyplot(
                model,
                img, (pytorch_result_, ),
                title='PyTorch',
                palette=model.PALETTE,
                opacity=0.5)
        # compare results
        np.testing.assert_allclose(
            pytorch_result.astype(np.float32) / num_classes,
            onnx_result.astype(np.float32) / num_classes,
            rtol=1e-5,
            atol=1e-5,
            err_msg='The outputs are different between Pytorch and ONNX')
        print('The outputs are same between Pytorch and ONNX')

    if norm_cfg is not None:
        print("Prepending BatchNorm layer to ONNX as data normalization...")
        mean = norm_cfg['mean']
        std = norm_cfg['std']
        i_n = m.graph.input[0]
        if (
            i_n.type.tensor_type.shape.dim[1].dim_value != len(mean)
            or i_n.type.tensor_type.shape.dim[1].dim_value != len(std)
        ):
            raise ValueError(
                f"--pixel-bias-value ({mean}) and --pixel-scale-value "
                f"({std}) should be same as input dimension: "
                f"{i_n.type.tensor_type.shape.dim[1].dim_value}"
            )
        norm_bn_bias = [-1 * cm / cs + 128. / cs for cm, cs in zip(mean, std)]
        norm_bn_scale = [1 / cs for cs in std]
        other.add_bias_scale_bn_after(
            m.graph, i_n.name, norm_bn_bias, norm_bn_scale
        )
        m = other.polish_model(m)
        bn_outf = os.path.splitext(output_file)[0] + "_bn_prepended.onnx"
        onnx.save(m, bn_outf)
        print(f"BN-Prepended ONNX saved to {bn_outf}")

    return


def parse_args():
    parser = argparse.ArgumentParser(description='Convert MMSeg to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file', default=None)
    parser.add_argument(
        '--input-img', type=str, help='Images for input', default=None)
    parser.add_argument(
        '--show',
        action='store_true',
        help='show onnx graph and segmentation results')
    parser.add_argument(
        '--verify', action='store_true', help='verify the onnx model')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=None,
        help='input image height and width.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--normalization-in-onnx',
        action='store_true',
        help='Prepend BatchNorm layer to onnx model as a role of data '
        'normalization according to the mean and std value in the given'
        'cfg file.'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    assert args.opset_version == 11, "kneron_toolchain currently only supports opset 11"

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.model.pretrained = None

    test_mode = cfg.model.test_cfg.mode

    if args.shape is None:
        if test_mode == 'slide':
            crop_size = cfg.model.test_cfg['crop_size']
            input_shape = (1, 3, crop_size[1], crop_size[0])
        else:
            img_scale = cfg.test_pipeline[1]['img_scale']
            input_shape = (1, 3, img_scale[1], img_scale[0])
    else:
        if test_mode == 'slide':
            warnings.warn(
                "We suggest you NOT assigning shape when exporting "
                "slide-mode models. Assigning shape to slide-mode models "
                "may result in unexpected results. To see which mode the "
                "model is using, check cfg.model.test_cfg.mode, which "
                "should be either 'whole' or 'slide'."
            )
        if len(args.shape) == 1:
            input_shape = (1, 3, args.shape[0], args.shape[0])
        elif len(args.shape) == 2:
            input_shape = (
                1,
                3,
            ) + tuple(args.shape)
        else:
            raise ValueError('invalid input shape')

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    segmentor = build_segmentor(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    # convert SyncBN to BN
    segmentor = _convert_batchnorm(segmentor)

    if args.checkpoint:
        checkpoint = load_checkpoint(
            segmentor, args.checkpoint, map_location='cpu')
        segmentor.CLASSES = checkpoint['meta']['CLASSES']
        segmentor.PALETTE = checkpoint['meta']['PALETTE']

    # read input or create dummpy input
    if args.input_img is not None:
        preprocess_shape = (input_shape[2], input_shape[3])
        img = _prepare_input_img(
            args.input_img,
            cfg.data.test.pipeline,
            shape=preprocess_shape)
    else:
        img = _demo_mm_inputs(input_shape)

    if args.normalization_in_onnx:
        norm_cfg = _parse_normalize_cfg(cfg.test_pipeline)
    else:
        norm_cfg = None
    # convert model to onnx file
    pytorch2onnx(
        segmentor,
        img,
        norm_cfg=norm_cfg,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify,
    )
