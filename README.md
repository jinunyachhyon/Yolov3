# YOLOv3 Object Detection
This repository implements YOLOv3 (You Only Look Once version 3) for object detection using PyTorch. <br>
The link to original paper - **YOLOv3: An Incremental Improvement** ([Link](https://arxiv.org/abs/1804.02767))


## Files

- `inference.py`: Python script for performing object detection on images.
- `inference_video.py`: Python script for object detection on videos.
- `Yolov3_epoch80.pth`: PyTorch model file using LFS containing pre-trained weights for YOLOv3 at epoch 80.
- `config.py`: Configuration file for YOLOv3 model parameters.
- `dataset.py`: Module for handling datasets in YOLOv3.
- `loss.py`: Module defining the loss function for YOLOv3.
- `model.py`: Implementation of the YOLOv3 model architecture.
- `stitch_im2vdo.py`: Script for stitching images into a video after inference.
- `train.py`: Script for training the YOLOv3 model.
- `utils.py`: Utility functions used in YOLOv3.


## Usage

### Object Detection on Images
To perform object detection on an image, use `inference.py`:

```bash
python inference.py --model path/to/model --test_image path/to/image
```

### Training
To train the YOLOv3 model, execute `train.py`:

```bash
python train.py
```

### Testing
To test the YOLOv3 model, execute `test.py`:

```bash
python test.py --model path/to/model
```


## Installation

### Installation Steps
Clone this repository:

```bash
git clone https://github.com/jinunyachhyon/Yolov3.git
cd Yolov3
```

### Install dependencies:

``` bash
pip install -r requirements.txt
```

### Use pre-trained weights:
Instead of training, you can use the pre-trained weight: `Yolov3_epoch80.pth` or `Yolov3_epoch50.pth`.

* `Yolov3_epoch80.pth` : Trained on test.csv (4951 unique values) performs well on test-set images.
* `Yolov3_epoch50.pth` : Trained on train.csv (16550 unique values) performs well on train-set images.


## Result

For `Yolov3_epoch80.pth` trained on test.csv (nearly 5k data), testing on seen dataset.

* Loss = 2.0037
* mAP = 0.837194


## Contributing
Contributions, issues, and feature requests are welcome. Feel free to open issues or pull requests for improvements or fixes.


## License
This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details.


## Acknowledgments
The YOLOv3 implementation is based on the original YOLOv3 paper by Joseph Redmon and Ali Farhadi. <br>
```
@misc{redmon2018yolov3,
      title={YOLOv3: An Incremental Improvement}, 
      author={Joseph Redmon and Ali Farhadi},
      year={2018},
      eprint={1804.02767},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

