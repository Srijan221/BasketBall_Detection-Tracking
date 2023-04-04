# Custom YOLOv7 Object Detector with YOLO Labels using PyTorch
Here, we will fine-tune/train our YOLOv7x and YOLOv7-tiny model for Basketball detection. We are going to use PyTorch for this. Current python version 3.8.10.

<h2> Data Gathering </h2>

Get the dataset for training, testing and validation in the dataset folder with images and labels in seperate folder.
I am running a Cleanup.py file to erase all the other labels from the labels.txt files of images for preventing any error that can come up while training due to mismatched class names.


<h2> Create yaml file  </h2>

As we have the YOLO labeled images, we will create a basketball.yaml file in the data directory which will be loaded while training.
Modify below line in the file `basketball.yaml` inside `data` folder for training more than one object and to change the paths of the train, val and test images.

```
train: ../yolov7/dataset/train/images
val: ../yolov7/dataset/valid/images
test: ../yolov7/dataset/test/images

nc: 1
names: ['basketball']
```

Since I am creating the model for only basketball detection. I have given the name `Basketball` and number_of_classes = 1

## Modifying the config file

Next, we have to modify the config file for specifying the model architecture and number of dense layers to use during training. 

When training YOLOv7x, modify the yolov7x.yaml file and replace `nc: 1` for specifying just one class. Same goes when training the YOLOv7-tiny

## Begin Training
Run the following commands in your PytTorch environment - 

Install dependendecies if any dependency is missing by using 

```
pip install -r requirements.txt
```
Download the YOLOv7 model weights from here [`yolov7x.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt), ['yolov7-tiny.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt)

Now we are ready for training our YOLOv7 models.

Run the following command to begin training
```
For YOLOv7x - 

python train.py --epochs 100 --workers 4 --device 0 --batch-size 4 --data data/basketball.yaml --img 640 640 --cfg cfg/training/yolov7x_basketball.yaml --weights 'yolov7x.pt' --name yolov7_basketball_multi_res --hyp data/hyp.scratch.custom.yaml --multi-scale

For YOLOv7-tiny - 

python train.py --epochs 100 --workers 4 --device 0 --batch-size 16 --data data/basketball.yaml --img 640 640 --cfg cfg/training/yolov7-tiny.yaml --weights 'yolov7-tiny.pt' --name yolov7_tiny_basketball_multi_res --hyp data/hyp.scratch.tiny.yaml

```

I have enabled multi-scaling of resolution of images, this means that the the size of the images will be varied every few batches.
In multi-resolution training, we need to provide the base resolution (say, 640×640). During training, the images will be resized to +-50% if this base resolution. So, for 640×640 images, the minimum resolution will be 320×320 and the maximum resolution will be 1280×1280. Generally, this helps to train a more robust model especially for cases when we have smaller objects, like this dataset. But we also need to train for longer as the dataset becomes much more difficult because of the varied sizes.


## Test the fine-tuned model and run detections
After training, new model will be available in a new directory with path `runs/train/yolov7_basketball_multi_res/weights` as `best.pt`. Now we can test this model on the test dataset and run detections by executing the following command - 

```
For Testing - 
python test.py --weights runs/train/yolov7_basketball_multi_res/weights/best.pt --task test --data data/basketball.yaml

To run detections on a webcam source - 
python detect.py --source 0 --weights runs/train/yolov7_basketball_multi_res/weights/best.pt --view-img
```

As the model is fine-tuned on the Basketball dataset, it is ready to be implemented with any type of tracking algorithm.