# tiny-yolov4-darknet-object-detection-custom-Training

# yolov4-darknet-object-detection-custom-Training

![download](https://user-images.githubusercontent.com/78655282/128880791-374b1bee-94cb-4ff0-98a4-1dc150a2ed0d.png)


YOLO stands for You Only Look Once. YOLO is a state-of-the-art, real-time object detection system. It was developed by Joseph Redmon. It is a real-time object recognition system that can recognize multiple objects in a single frame.

##How YOLO works
YOLO uses a totally different approach than other previous detection systems. It applies a single neural network to the full image. This network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities.
The basic idea of YOLO is exhibited in the figure below. YOLO divides the input image into an S × S grid and each grid cell is responsible for predicting the object centered in that grid cell.


![0_Okuwq93g3v13CShN](https://user-images.githubusercontent.com/78655282/128882771-c7c9b824-f8d9-4a82-96c3-9e42d0584a50.jpg)


Each grid cell predicts B bounding boxes and confidence scores for those boxes. These confidence scores reflect how confident the model is that the box contains an object and also how accurate it thinks the box is that it predicts.


![0_IhbtJNWpPG1PgTRk](https://user-images.githubusercontent.com/78655282/128882941-3cd7a591-96a1-463e-95c9-c5aad67cb251.png)


## What is YOLOv4-tiny?
YOLOv4-tiny is the compressed version of YOLOv4. It is proposed based on YOLOv4 to make the network structure simpler and reduce parameters so that it becomes feasible for developing on mobile and embedded devices.
We can use YOLOv4-tiny for much faster training and much faster detection. It has only two YOLO heads as opposed to three in YOLOv4 and it has been trained from 29 pre-trained convolutional layers as opposed to YOLOv4 which has been trained from 137 pre-trained convolutional layers.
The FPS (Frames Per Second) in YOLOv4-tiny is approximately eight times that of YOLOv4. However, the accuracy for YOLOv4-tiny is 2/3rds that of YOLOv4 when tested on the MS COCO dataset.
The YOLOv4-tiny model achieves 22.0% AP (42.0% AP50) at a speed of 443 FPS on RTX 2080Ti, while by using TensorRT, batch size = 4 and FP16-precision the YOLOv4-tiny achieves 1774 FPS.
For real-time object detection, YOLOv4-tiny is the better option when compared with YOLOv4 as faster inference time is more important than precision or accuracy when working with a real-time object detection environment.

![1_m2N0pah1h1fYKlkAYGlsLQ](https://user-images.githubusercontent.com/78655282/128883815-3f1618fb-9636-4a85-b0e5-1876f5f5a075.png)

## For more information, you can use the following link.

- [You Only Look Once:Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)

- [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/pdf/2004.10934.pdf)


## How to train a YOLO model on our data?To do this, we use Darknet and run the model on the database and finally check the results.

### To train the YOLOv4 model, we must perform the following steps in order

- Preparing a custom Scaled YOLOv4 Dataset
  - To train your object detector, you will need to bring labeled image data that teaches the model what it needs to detect.
  - If your data contains labels, you can use this link [this link](https://roboflow.com/formats/yolo-darknet-txt) to convert your data labels to usable labels in YOLO Darknet.
  - Also, if your data does not have a label, you can use link [this link](https://github.com/tzutalin/labelImg) to label your data. In the given link, there is an app by which you can easily label the images and prepare for YOLO training.

- Open the YOLOv4 Darknet notebook at the top and follow the steps mentioned in the notebook. This notebook is prepared to find the location of the screws in the image. An example of the image and output is as follows.
