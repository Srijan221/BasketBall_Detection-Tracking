# BasketBall_Detection-Tracking

This repo uses official implementations of [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://github.com/WongKinYiu/yolov7) and [Simple Online and Realtime Tracking with a Deep Association Metric (Deep SORT)](https://github.com/nwojke/deep_sort)  to detect Basketball from videos and then track Basketball to keep detecting even if it goes out of frame for some secs.

Here is a implemtation of the whole pipeline - 

![basketball_tracker-gif](https://user-images.githubusercontent.com/69648635/229661446-6f088b4d-21e7-487e-864b-97e86ec9788d.gif)


## Steps to run:
First we need to train our YOLOv7-tiny for Basketball class and fine-tune the pretrained weights of YOLO.
To do so, go to the `Training_Models/YOLOv7_detector` by clicking [here](https://github.com/Srijan221/BasketBall_Detection-Tracking/blob/main/Training%20Models/YOLOv7_detector/Readme.md) and follow the instructions.

If you want to train MobileNetV2 SSD model, you can follow the instructions from here and get started - [Custom Train MobileNetSSD](https://github.com/Srijan221/BasketBall_Detection-Tracking/blob/main/Training%20Models/SSD_object_detector/BasketBall_detection/Readme.md)

Once done with the training, copy the best.pt model to the `weights` directory. There is already a best_yolov7-tiny.pt which is trained for 100 epochs on the [UC_Berkely_Basketball_dataset](https://universe.roboflow.com/uc-berkely-w210-tracer/tracer-basketball) and achivies a mAP of 95%.

To use this, run the following command in your PyTorch Environment to install the requirements first - 

```
pip install -r requirements.txt
```

After installing the requirements, run the following command to run the detector+tracker for your video - 

```
python main.py --show_live --draw_line --count_objects --input /path/to/your/video --output /path/to/save/your/video --weights /path/to/your/model
```
```
args - 
--show_live: to show live tracking
--count_objects: count number of objects detected
--draw_line: draw trajectory of the Basketball
--input: file path of your video
--output: file path where you want to save your video
--weights: path to .pt file of your model
--verbose: print details on the screen
--graph: plot and save graph of the (x_center, y_center) coordinates of the Basketball vs time, allowed values 2D,3D
```

For example, to run the existing model execute - 

```
python main.py --show_live --draw_line --count_objects --graph 3D

```

# Troubleshooting
This code works **perfectly** with `python==3.8.10, tensorflow==2.8.0, torch== 1.8.0, sklearn==0.24.2` on local **Ubuntu: CPU** as well as **Nvidia 3060 single core GPU** as of `04/04/2023`.

If you find anytrouble, please raise an issue.


# References
1. [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://github.com/WongKinYiu/yolov7)
2. [Simple Online and Realtime Tracking with a Deep Association Metric (Deep SORT)](https://github.com/nwojke/deep_sort)
3. [UC_Berkely_Basketball_dataset](https://universe.roboflow.com/uc-berkely-w210-tracer/tracer-basketball)
4. 