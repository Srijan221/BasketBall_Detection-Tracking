# Custom Object Detector for YOLO Images using Tensorflow
Here, we will create SSD-MobileNet-V2 model for candy detection. We are going to use tensorflow-gpu 2.2 for this. I am using python version 3.8.10.

<h2> Data Gathering </h2>


In `config.yaml`, mention your combined directory of training+testing images and label(.txt) files.

Also change the amount of images you want to split your dataset into ( `default - 0.2`)


<h2> Create lable map file  </h2>

Modify below line in the file `object-detection.pbxt` inside `training` folder for training more than one object

```
item {
  id: 1
  name: 'candy'
}

```

Since I am creating the model for only candy object detection. I have given name `candy`. Modify it at your own will.

## To get the exported model directly into the exported directory
Run the following command in your Tensorflow environment - 

Install dependendecies if any dependency is missing

```
pip install pycocotools
pip install scipy
pip install dataclasses
pip install pyyaml

pip install tf_slim
pip install pillow
pip install lxml
pip install jupyter
pip install matplotlib

```
Run the following to create tf.records file for train and test dataset in the `data` folder and begin training.

```
bash script.bash

```
After training new model directory named `candy-model` will be created inside the exported-model directory automatically.

If in any case your total_loss starts to shoot up while training, you can stop the run and export the last trained_checkpoints by running following command - 

```
python exporter_main_v2.py --input_type image_tensor --pipeline_config_path ./ssd_mobilenet_v2_320x320_coco17_tpu-8/pipeline.config --trained_checkpoint_dir ./trained-checkpoint --output_directory exported-model/candy-model

```