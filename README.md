# Introduction
The purpose of this project is trying to do multiple objects tracking using YOLOv3 with DeepSORT, and make the people tracking more robust. Thanks for these projects, this project now can support YOLOv3.

  https://github.com/nwojke/deep_sort

  https://github.com/qqwweee/keras-yolo3

  https://github.com/Qidian213/deep_sort_yolov3


# Some excellent related work
   1. https://github.com/xingyizhou/CenterTrack

   2. https://github.com/phil-bergmann/tracking_wo_bnw

   3. https://github.com/Zhongdao/Towards-Realtime-MOT

   4. https://github.com/ifzhang/FairMOT

   5. https://github.com/pjl1995/CTracker



# Quick Start

1. Download YOLOv3 or tiny_yolov3 weights and cfg from [YOLO website](http://pjreddie.com/darknet/yolo/), and put it into model_data folder.

2. Put the video into the test_video folder, and modify the video_file name in `demo.py`.

3. Run YOLO_DEEP_SORT with command :
   ```
   python demo.py
   ```


# Training the model

To train the deep association metric model on your datasets you can reference to [cosine_metric_learning](https://github.com/nwojke/cosine_metric_learning) approach which is provided as a separate repository.


# Result Video

1. Result video on MOT16-13

[![result video1](http://img.youtube.com/vi/m3f50s5XJs4/0.jpg)](https://www.youtube.com/watch?v=m3f50s5XJs4 "multiple Objects Tracking1")



2. Result video on Taiwan street

[![result video2](http://img.youtube.com/vi/pYXLm9WRTwk/0.jpg)](https://www.youtube.com/watch?v=pYXLm9WRTwk "multiple Objects Tracking2")
