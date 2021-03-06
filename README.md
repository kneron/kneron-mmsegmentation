# Kneron AI Training/Deployment Platform (mmsegmentation-based)


## Introduction

  [kneron-mmsegmentation](https://github.com/kneron/kneron-mmsegmentation) is a platform built upon the well-known [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) for mmsegmentation. If you are looking for original mmsegmentation document, please visit [mmsegmentation docs](https://mmsegmentation.readthedocs.io/en/latest/) for detailed mmsegmentation usage.

  In this repository, we provide an end-to-end training/deployment flow to realize on Kneron's AI accelerators: 

  1. **Training/Evalulation:**
      - Modified model configuration file and verified for Kneron hardware platform 
      - Please see [Overview of Benchmark and Model Zoo](#Overview-of-Benchmark-and-Model-Zoo) for Kneron-Verified model list
  2. **Converting to ONNX:** 
      - tools/pytorch2onnx_kneron.py (beta)
      - Export *optimized* and *Kneron-toolchain supported* onnx
          - Automatically modify model for arbitrary data normalization preprocess
  3. **Evaluation**
      - tools/test_kneron.py (beta)
      - Evaluate the model with *pytorch checkpoint, onnx, and kneron-nef*
  4. **Testing**
      - inference_kn (beta)
      - Verify the converted [NEF](http://doc.kneron.com/docs/#toolchain/manual/#5-nef-workflow) model on Kneron USB accelerator with this API
  5. **Converting Kneron-NEF:** (toolchain feature)
     - Convert the trained pytorch model to [Kneron-NEF](http://doc.kneron.com/docs/#toolchain/manual/#5-nef-workflow) model, which could be used on Kneron hardware platform.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog

N/A

## Overview of Benchmark and Kneron Model Zoo

| Backbone | Crop Size | Mem (GB) | mIoU | Config | Download |
|:--------:|:---------:|:--------:|:----:|:------:|:--------:|
| STDC 1   | 512x1024  | 7.15     | 69.29|[config](https://github.com/kneron/kneron-mmsegmentation/tree/master/configs/stdc/kn_stdc1_in1k-pre_512x1024_80k_cityscapes.py)|[model](https://github.com/kneron/Model_Zoo/blob/main/mmsegmentation/stdc_1/latest.zip)

NOTE: The performance may slightly differ from the original implementation since the input size is smaller.

## Installation
- Please refer to the Step 1 of [docs_kneron/stdc_step_by_step.md#step-1-environment](docs_kneron/stdc_step_by_step.md) for installation.
- Please refer to [Kneron PLUS - Python: Installation](http://doc.kneron.com/docs/#plus_python/introduction/install_dependency/) for the environment setup for Kneron USB accelerator.

## Getting Started
### Tutorial - Kneron Edition
- [STDC-Seg: Step-By-Step](docs_kneron/stdc_step_by_step.md): A tutorial for users to get started easily. To see detailed documents, please see below.

### Documents - Kneron Edition
- [Kneron ONNX Export] (under development)
- [Kneron Inference] (under development)
- [Kneron Toolchain Step-By-Step (YOLOv3)](http://doc.kneron.com/docs/#toolchain/yolo_example/)
- [Kneron Toolchain Manual](http://doc.kneron.com/docs/#toolchain/manual/#0-overview)

### Original mmsegmentation Documents
- [Original mmsegmentation getting started](https://github.com/open-mmlab/mmsegmentation#getting-started): It is recommended to read the original mmsegmentation getting started documents for other mmsegmentation operations.
- [Original mmsegmentation readthedoc](https://mmsegmentation.readthedocs.io/en/latest/): Original mmsegmentation documents.

## Contributing
[kneron-mmsegmentation](https://github.com/kneron/kneron-mmsegmentation) a platform built upon [OpenMMLab-mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

- For issues regarding to the original [mmsegmentation](https://github.com/open-mmlab/mmsegmentation):
We appreciate all contributions to improve [OpenMMLab-mmsegmentation](https://github.com/open-mmlab/mmsegmentation). Ongoing projects can be found in out [GitHub Projects](https://github.com/open-mmlab/mmsegmentation/projects). Welcome community users to participate in these projects. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

- For issues regarding to this repository [kneron-mmsegmentation](https://github.com/kneron/kneron-mmsegmentation): Welcome to leave the comment or submit pull requests here to improve kneron-mmsegmentation


## Related Projects
- [kneron-mmdetection](https://github.com/kneron/kneron-mmdetection): Kneron training/deployment platform on [OpenMMLab - mmdetection](https://github.com/open-mmlab/mmdetection) object detection toolbox
