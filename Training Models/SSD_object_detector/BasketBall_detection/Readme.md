# Custom MobileNetSSD Object Detector with YOLO Labels using Tensorflow
Here, we will train SSD-MobileNet-V2 model for Basketball detection. We are going to use tensorflow-gpu 2.2 for this. Current python version 3.8.10.

<h2> Data Gathering </h2>

In `config.yaml`, mention your combined directory of training+testing YOLO images and label(.txt) files.

Also change the percentage of total images you want to split your dataset into ( `default - 0.2`)

As we have the YOLO labeled images, we have to first convert this in Tf-Record so that the tensorflow can detect these labels and use them for training the MobileNet-SSD model.


<h2> Create lable map file  </h2>

Modify below line in the file `object-detection.pbxt` inside `training` folder for training more than one object

```
item {
  id: 1
  name: 'Basketball'
}

```

Since I am creating the model for only basketball detection. I have given the name `Basketball` and id: 1.

## To get the exported model directly into the exported directory
Run the following command in your Tensorflow environment - 

Install dependendecies if any dependency is missing using 

```
pip install -r requirements.txt
```

Run the following to create tf.records file for train and test dataset in the `data` folder.

```
bash script.bash

```
Training of the MobileNetSSD-v2 will start after creation of tf.records.

After training new model directory named `basketball-model` will be created inside the exported-model directory automatically.

If in any case your total_loss starts to shoot up while training, you can stop the run and export the last trained_checkpoints by running following command - 

```
python exporter_main_v2.py --input_type image_tensor --pipeline_config_path ./ssd_mobilenet_v2_320x320_coco17_tpu-8/pipeline.config --trained_checkpoint_dir ./trained-checkpoint --output_directory exported-model/candy-model

```

