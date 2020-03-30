# DeepLetters

Deep learning based text detection and recognition using tensorflow.

DeepLetters consists of two convolutional networks.
* [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) for text detection
* [Convolutional Recurrent Neural Network](https://arxiv.org/abs/1507.05717) for text recognition

## results

![result1](results/result.jpg)

## How to use

```csh
$ python deep_letters.py --input <input image or video> --detection_model_path <detection_model_pb> --detection_th <th> --recognition_model_path <recognition_model.pth>
```

Download detection model file (pb file) from [GoogleDrive](https://drive.google.com/open?id=1qXlfxkDdvW3dFS6NPZdC3VOmhb9DUsSf).

Clone [crnn.pytorch](https://github.com/meijieru/crnn.pytorch) repository and place it on same level with DeepLetters repository. (DeepLetters use crnn.pytorch to recognize texts in detection results)
