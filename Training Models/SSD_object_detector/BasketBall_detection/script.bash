#!/bin/bash
function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}
eval $(parse_yaml config.yaml)


echo "Converting all your training images .txt file to .xml"
python yolo_to_xml.py
echo "Now, splitting the dataset into train and test"
python train_test_split.py
echo "Converting .xml to .csv"
python xml_to_csv.py
cd models/research/
# compiles the proto buffers
protoc object_detection/protos/*.proto --python_out=.
# exports PYTHONPATH environment var with research and slim paths
cd ..
cd ..
echo "Converting .csv to tfrecord file!"

python generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record --image_dir=images/train 
python generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record --image_dir=images/test

echo "TRAINING BEGINS--------------"

python model_main_tf2.py --pipeline_config_path=ssd_mobilenet_v2_320x320_coco17_tpu-8/pipeline.config --model_dir=trained-checkpoint --alsologtostderr --num_train_steps=$step_size_from_yaml --sample_1_of_n_eval_examples=1 --num_eval_steps=1

echo "TRAINING ENDS--------------"

python exporter_main_v2.py --input_type image_tensor --pipeline_config_path ./ssd_mobilenet_v2_320x320_coco17_tpu-8/pipeline.config --trained_checkpoint_dir ./trained-checkpoint --output_directory exported-model/basketball-model

echo "Model exported successfully!!!"
