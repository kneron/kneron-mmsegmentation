# All modification made by Kneron Corp.: Copyright (c) 2022 Kneron Corp.
# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Any, Iterable, Union
from os import path as osp

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import BaseModule, auto_fp16
from mmseg.core import get_classes, get_palette
from mmseg.ops import resize


class BaseSegmentor(BaseModule, metaclass=ABCMeta):
    """Base class for segmentors."""

    def __init__(self, init_cfg=None):
        super(BaseSegmentor, self).__init__(init_cfg)
        self.fp16_enabled = False

    @property
    def with_neck(self):
        """bool: whether the segmentor has neck"""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_auxiliary_head(self):
        """bool: whether the segmentor has auxiliary head"""
        return hasattr(self,
                       'auxiliary_head') and self.auxiliary_head is not None

    @property
    def with_decode_head(self):
        """bool: whether the segmentor has decode head"""
        return hasattr(self, 'decode_head') and self.decode_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        """Placeholder for extract features from images."""
        pass

    @abstractmethod
    def encode_decode(self, img, img_metas):
        """Placeholder for encode images with backbone and decode into a
        semantic segmentation map of the same size as input."""
        pass

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        """Placeholder for Forward function for training."""
        pass

    @abstractmethod
    def simple_test(self, img, img_meta, **kwargs):
        """Placeholder for single image test."""
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Placeholder for augmentation test."""
        pass

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got '
                                f'{type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) != '
                             f'num of image meta ({len(img_metas)})')
        # all images in the same aug batch all of the same ori_shape and pad
        # shape
        for img_meta in img_metas:
            ori_shapes = [_['ori_shape'] for _ in img_meta]
            assert all(shape == ori_shapes[0] for shape in ori_shapes)
            img_shapes = [_['img_shape'] for _ in img_meta]
            assert all(shape == img_shapes[0] for shape in img_shapes)
            pad_shapes = [_['pad_shape'] for _ in img_meta]
            assert all(shape == pad_shapes[0] for shape in pad_shapes)

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']))

        return outputs

    def val_step(self, data_batch, optimizer=None, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']))

        return outputs

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # If the loss_vars has different length, raise assertion error
        # to prevent GPUs from infinite waiting.
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()) + '\n')
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def show_result(self,
                    img,
                    result,
                    palette=None,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    opacity=0.5):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor): The semantic segmentation results to draw over
                `img`.
            palette (list[list[int]]] | np.ndarray | None): The palette of
                segmentation map. If None is given, random palette will be
                generated. Default: None
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.
            opacity(float): Opacity of painted segmentation map.
                Default 0.5.
                Must be in (0, 1] range.
        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        seg = result[0]
        if palette is None:
            if self.PALETTE is None:
                # Get random state before set seed,
                # and restore random state later.
                # It will prevent loss of randomness, as the palette
                # may be different in each iteration if not specified.
                # See: https://github.com/open-mmlab/mmdetection/issues/5844
                state = np.random.get_state()
                np.random.seed(42)
                # random palette
                palette = np.random.randint(
                    0, 255, size=(len(self.CLASSES), 3))
                np.random.set_state(state)
            else:
                palette = self.PALETTE
        palette = np.array(palette)
        assert palette.shape[0] == len(self.CLASSES)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        if show:
            mmcv.imshow(img, win_name, wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img


class ONNXRuntimeSegmentorKN(BaseSegmentor):

    def __init__(
            self,
            onnx_file: str,
            cfg: Any,
            device_id: Union[int, None] = 0):
        super(ONNXRuntimeSegmentorKN, self).__init__()
        import onnxruntime as ort

        # get the custom op path
        ort_custom_op_path = ''
        try:
            from mmcv.ops import get_onnxruntime_op_path
            ort_custom_op_path = get_onnxruntime_op_path()
        except (ImportError, ModuleNotFoundError):
            warnings.warn(
                'If input model has custom op from mmcv, you may '
                'have to build mmcv with ONNXRuntime from source.')
        session_options = ort.SessionOptions()
        # register custom op for onnxruntime
        if osp.exists(ort_custom_op_path):
            session_options.register_custom_ops_library(ort_custom_op_path)
        providers = ['CPUExecutionProvider']
        provider_options = [{}]
        is_cuda_available = (
            ort.get_device() == 'GPU' and torch.cuda.is_available()
        )
        if is_cuda_available:
            providers.insert(0, 'CUDAExecutionProvider')
            device_id = device_id or 0
            provider_options.insert(0, {'device_id': device_id})
        sess = ort.InferenceSession(
            onnx_file, session_options, providers, provider_options
        )
        self.sess = sess
        sess_inputs = sess.get_inputs()
        assert len(sess_inputs) == 1, "Only onnx with 1 input is supported"
        self.input_name = sess_inputs[0].name
        sess_outputs = sess.get_outputs()
        self.num_classes = sess_outputs[0].shape[1]
        assert len(sess_outputs) == 1, "Only onnx with 1 output is supported"
        self.output_name_list = [sess_outputs[0].name]
        self.cfg = cfg  # TODO: necessary?
        self.test_cfg = cfg.model.test_cfg
        self.test_mode = self.test_cfg.mode  # NOTE: either 'whole' or 'slide'
        self.is_cuda_available = is_cuda_available
        self.count_mat = None
        try:
            if 'test' in cfg.data:
                dataset_name = cfg.data.test['type']
            else:
                dataset_name = cfg.data.train['type']
            dataset_name = dataset_name.lower()[:-7]
            self.CLASSES = get_classes(dataset_name)
            self.PALETTE = get_palette(dataset_name)
        except (AttributeError, KeyError):
            warnings.warn(
                "Failed to fetch dataset name from config; no CLASSES "
                "and PALETTE for this ONNX model"
            )
        except ValueError:
            warnings.warn(
                "Failed to fetch CLASSES and PALETTE from dataset "
                f"{dataset_name}; no CLASSES and PALETTE for this "
                "ONNX MODEL."
            )

    def extract_feat(self, imgs):
        raise NotImplementedError('This method is not implemented.')

    def encode_decode(self, img, img_metas):
        raise NotImplementedError('This method is not implemented.')

    def forward_train(self, imgs, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')

    def forward_test(self, imgs, img_metas, **kwargs):
        return super().forward_test(imgs, img_metas[0].data, **kwargs)

    def simple_slide_inference(
            self,
            img: np.ndarray,
            img_meta: Union[Iterable, None] = None):
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        _, _, h_img, w_img = img.shape
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = np.zeros((1, num_classes, h_img, w_img), dtype=np.float32)
        # NOTE: count_mat should be invariant since
        #       input shape of kneron's onnx is fixed
        if self.count_mat is None:
            count_mat = np.zeros((1, 1, h_img, w_img), dtype=np.float32)
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.sess.run(
                    self.output_name_list,
                    {self.input_name: crop_img}
                )[0]
                preds += np.pad(
                    crop_seg_logit,
                    ([0, 0],
                     [0, 0],
                     [int(y1), int(preds.shape[2] - y2)],
                     [int(x1), int(preds.shape[3] - x2)]),
                )
                if self.count_mat is None:
                    count_mat[:, :, y1:y2, x1:x2] += 1
        if self.count_mat is None:
            assert (count_mat == 0).sum() == 0
            self.count_mat = count_mat
        preds /= self.count_mat
        return preds

    @property
    def module(self):
        return self

    @torch.no_grad()
    def simple_test(
            self,
            img: torch.Tensor,
            img_meta: Union[Iterable, None] = None,
            **kwargs) -> list:
        img = img.cpu().numpy()
        # NOTE: not using run_with_iobinding since some ort versions
        #       generate wrong results when inferencing with CUDA
        if self.test_mode == 'slide':
            seg_pred = self.simple_slide_inference(img, img_meta)
        else:
            seg_pred = self.sess.run(
                self.output_name_list, {self.input_name: img}
            )[0]
        if img_meta is not None:
            ori_shape = img_meta[0]['ori_shape']
            if not (ori_shape[0] == seg_pred.shape[-2]
                    and ori_shape[1] == seg_pred.shape[-1]):
                seg_pred = torch.from_numpy(seg_pred).float()
                seg_pred = resize(
                    seg_pred, size=tuple(ori_shape[:2]), mode='bilinear')
                seg_pred = seg_pred.numpy()
        elif img.shape[2:] != seg_pred.shape[2:]:
            seg_pred = torch.from_numpy(seg_pred).float()
            seg_pred = resize(
                seg_pred, size=(img.shape[3], img.shape[2]), mode='bilinear')
            seg_pred = seg_pred.numpy()
        seg_pred = seg_pred.argmax(1)
        return list(seg_pred)

    def aug_test(self, imgs, img_metas, **kwargs):
        raise NotImplementedError('This method is not implemented.')
