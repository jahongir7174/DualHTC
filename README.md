[HTC++](https://arxiv.org/abs/1901.07518) implementation using PyTorch (supports `MOSAIC` and `MixUp`)

#### Train (default configuration is for [COCO dataset](https://cocodataset.org/#home))

* Install [mmdet](https://github.com/jahongir7174/CBNetV2) toolbox
* Register your dataset to `utils/dataset.py`, see `DAMAGEDataset`
* See `nets/exp01.py` for using `MOSAIC` and `MixUp` data pipeline
* See `nets/exp02.py` for using `RandomAugment` data pipeline
* Run `bash ./main.sh ./nets/exp01.py $ --train` for training, `$` is number of GPUs

#### Reference

* https://github.com/ultralytics/yolov5
* https://github.com/open-mmlab/mmdetection
