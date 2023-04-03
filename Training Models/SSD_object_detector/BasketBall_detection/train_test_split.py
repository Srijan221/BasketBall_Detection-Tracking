import os
from glob import glob
import shutil
from sklearn.model_selection import train_test_split

import yaml
with open (r'/home/srijan/Downloads/BasketBall_detection/config.yaml') as file:
  config = yaml.load(file,Loader=yaml.FullLoader)
  train_test_split_percentage = config['train_test_split']
#getting list of images
image_files = glob("images/yolo_images_with_labels/*.jpg")

#replacing the extension
images = [name.replace(".jpg","") for name in image_files]
#splitting the dataset
train_names, test_names = train_test_split(images, test_size=float(train_test_split_percentage))

def batch_move_files(file_list, source_path, destination_path):
    for file in file_list:
        #extracting only the name of the file and concatenating with extenions
        # print(file)
        image = file.split('/')[2] + '.jpg'
        xml = file.split('/')[2] + '.xml'
        shutil.move(os.path.join(source_path, image), destination_path)
        shutil.move(os.path.join(source_path, xml), destination_path)
    return

source_dir = "images/yolo_images_with_labels/"
test_dir = "images/test"
train_dir = "images/train"
batch_move_files(train_names, source_dir, train_dir)
batch_move_files(test_names, source_dir, test_dir)