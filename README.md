# Video Classification With CNNs: Using The Codec As A Spatio-Temporal Activity Sensor

## Introduction

This repository contains our public tool for selective decoding and flow approximation (MVX) and our temporal stream 3D-CNN classifier. Please find the details in our paper:

> *Video Classification With CNNs: Using The Codec As A Spatio-Temporal Activity Sensor*, 
> Chadha A., Abbas A., Andreopoulos Y., 
> [[arXiv](https://arxiv.org/abs/1710.05112)]

If you use the tools provided in this repository, please cite our work:
```
@article{chadha2017video,
  title={Video Classification With CNNs: Using The Codec As A Spatio-Temporal Activity Sensor},
  author={Chadha, Aaron and Abbas, Alhabib and Andreopoulos, Yiannis},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2017},
  publisher={IEEE}
}
```

## Prerequisites

In order to run thie code you will need:

1. Python 2.7 
1. TensorFlow (tested with TensorFlow 1.1.0)

## Generating Approximated Flow Inputs

Compile the tool MVX using:

```
g++ *.cpp
```

Make sure you transcode your input to be encoded using the GBR color format, e.g:

```
ffmpeg -i video.avi
```

To extract the approximated flow along with RGB texture at active regions within frames, use:

```
./mvx -i video.mp4 -w 1 -r 10 -t 0 --rgb out.rgb --mv out.mv
```

This will output the flow and texture to out.mv and out.rgb respectively.

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



