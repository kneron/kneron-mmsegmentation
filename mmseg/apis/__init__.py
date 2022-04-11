# All modification made by Kneron Corp.: Copyright (c) 2022 Kneron Corp.
# Copyright (c) OpenMMLab. All rights reserved.
from .inference import (
    inference_segmentor,
    inference_segmentor_kn,
    init_segmentor,
    init_segmentor_kn,
    show_result_pyplot,
)
from .test import multi_gpu_test, single_gpu_test
from .train import (get_root_logger, init_random_seed, set_random_seed,
                    train_segmentor)

__all__ = [
    'get_root_logger', 'set_random_seed', 'train_segmentor',
    'init_segmentor', 'init_segmentor_kn', 'inference_segmentor',
    'inference_segmentor_kn', 'multi_gpu_test', 'single_gpu_test',
    'show_result_pyplot', 'init_random_seed'
]
