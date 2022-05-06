# EC523 Project: Slicing Aided Hyper Inference

Team member: 
1. Wangyi Chen timchen@bu.edu
2. Zhiyuan Liu lzy2022@bu.edu
3. Tingru Lian tinalian@bu.edu
4. Zhangchi Lu zhchlu@bu.edu


 

Inspired by “Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection” [1], our task is to create a generic slicing aided inference and fine tuning pipeline for small object detection Fig. Because the dataset such as ImageNet, Pascal VOC12, MS COCO are mostly based on low-resolution images therefore, they do not perform well on high-resolution images. The need is to detect small detect objects on the high-resolution images. 
 ![image](https://user-images.githubusercontent.com/90277008/162350186-1f1bba0d-fee3-47a4-abce-a8827cd3d43d.png)

Figure 1. Detection of small objects and inference on large images

Slicing Aid Fine-tuning slices the images into overlapped patches. Object detection frameworks, such as YOLOv5, perform the best on low resolution images as they are mostly trained on ImageNet and COCO, which are low-resolution datasets. A possible consequence of Slicing Aid Fine-tuning is that it would create overlapped anchors, in which the same objects are identified in both original and sliced images. Slicing Aid Hyper Inference aims to resolve this issue. After conducting Slicing Aid Fine-tuning, the overlapped identification would be merged back into original size using Non-maximum Suppression (NMS).

<img width="337" alt="Screen Shot 2022-04-07 at 22 19 15" src="https://user-images.githubusercontent.com/90277008/162350308-c2d8d389-7f50-45e1-8295-1b764142fadc.png">

Figure 2.  Slicing aided fine-tuning (top) and slicing aided hyper inference (bottom) methods.

[1]https://arxiv.org/pdf/2202.06934v2.pdf

## Installation
This project's code is completely using Python. Therefore it can be run on any python code editor such as Google Colab, Jupyter Notebook, etc. 
There are the steps to replicate the result:
1. Download yolov5.py for testing performance of yolov5 and save annotatinos. 
2. Download sahi.py for SAHI model
3. sahi_yolo_test.ipynb for testing SAHI and save annotatinos. 
4. Open review_object_detection_metrics-main for calculating MAP
5. calculate MAP for both YOLOv5 and YOLOv5 with SAHI
