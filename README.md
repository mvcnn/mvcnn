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
1. AVLib (https://github.com/libav/libav)
2. Python 2.7 
3. TensorFlow (tested with TensorFlow 1.1.0)

## Compiling MVX And Preparing Video Inputs

The MVX tool takes an x264 bitstream as input and outputs the approximated flow 
and selectively decoded frames.

Install dependencies, if you are using a linux machine run:

```
apt-get update
apt-get install libav-tools libavutil-dev ffmpeg libboost-all-dev
```

Compile MVX using the provided Makefile:

```
make
```

Transcode your input video to be encoded using the GBR colorspace:

```
ffmpeg -y -i sample.mp4 -c:v libx264rgb -b:v 512k -bf 0 -pix_fmt rgb24  -r 25 -strict -2 gbr_sample.mp4
```

To extract the approximated flow along with GBR texture at active regions within frames, use:

```
./mvx -w 1 -r 10 -t 0 --rgb out.rgb --mv out.mv gbr_sample.mp4
```

This outputs the motion vector frames to "out.mv" and selected RGB texture to "out.rgb". The script "load_mv.m" is included to read the extracted motion vector frames (requires Matlab). -w specifies that RGB texture should be written every 1 frame (i.e. every frame) -r specifies that all texture should retrieved every 10 frames, and t is the motion vector activity threshold at which RGB texture is written for frames in between reference frames (write if dx^2 + dy^2 > threshold, where dx and dy are the estimated displacements along the x and y axis respectively). Please have a look at our paper to have a better idea of how these parameters affect the classifier performance.


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
python convert_UCF_bin.py --data_dir=<path to motion vector data dir --out_dir=<path to output dir> --path_to_file_list=<path to test split 1 file list (ucfTrainTestlist/testlist01.txt)>
```
2. Test on the converted bins:
```
python test.py --data_dir=<path to converted bins> --save_dir=<path to saved checkpoints/summaries>
```



