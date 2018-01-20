# Video Classification With CNNs: Using The Codec As A Spatio-Temporal Activity Sensor

This repository contains TensorFlow implementation of our temporal stream 3D-CNN with motion vector inputs, as reported in our paper:

> *Video Classification With CNNs: Using The Codec As A Spatio-Temporal Activity Sensor*, 
> Chadha A., Abbas A., Andreopoulos Y., 
> [[arXiv](https://arxiv.org/abs/1710.05112)]

## Prerequisites

In order to run thie code you will need:

1. Python 2.7 
1. TensorFlow (tested with TensorFlow 1.1.0)

## Training
In order to train our model on UCF-101 (split 1) on a single GPU:

1. Convert the train motion vector bins by running:
```
python convert_UCF_bin.py --data_dir=<path to motion vector data dir (ucf_8x8_w10_r1_bin)> --out_dir=<path to output dir> --path_to_file_list=<path to train split 1 file list (ucfTrainTestlist/trainlist01.txt)>
```
2. Train on the converted bins:
```
python train.py --data_dir=<path to converted bins> --save_dir=<path to save checkpoints/summaries>
```

Summary graphs can be viewed using TensorBoard, using the following command in Terminal:
```
tensorboard --logdir=<path to saved checkpoints/summaries>
```

## Testing
In order to test our model on UCF-101 (split 1) on a single GPU:

1. Convert the test motion vector bins by running:
```
python convert_UCF_bin.py --data_dir=<path to motion vector data dir (ucf_8x8_w10_r1_bin)> --out_dir=<path to output dir> --path_to_file_list=<path to test split 1 file list (ucfTrainTestlist/testlist01.txt)>
```
2. Test on the converted bins:
```
python test.py --data_dir=<path to converted bins> --save_dir=<path to saved checkpoints/summaries>
```



