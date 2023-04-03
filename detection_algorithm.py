'''
This code is taken from the official YOLOV7 repository and modified to bind with the tracking algotithm
'''

import cv2
import torch
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import letterbox, np
from utils.general import check_img_size, non_max_suppression, apply_classifier,scale_coords, xyxy2xywh
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier,TracedModel


class Detector:
    def __init__(self, conf_thres:float = 0.1, iou_thresh:float = 0.45, agnostic_nms:bool = False, save_conf:bool = False, classes:list = None):

        self.device = select_device("0" if torch.cuda.is_available() else 'cpu')
        self.conf_thres = conf_thres
        self.iou_thres = iou_thresh
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.save_conf = save_conf


    def load_model(self, weights:str, img_size:int = 640, trace:bool = True):

        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(img_size, s=self.stride)  # check img_size

        if trace:
            self.model = TracedModel(self.model, self.device, img_size) # this creates a TorchScript for fast loading of model when running

        if self.half:
            self.model.half()  # to FP1
        
        # Run inference for CUDA just once
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

         # Get names and colors of Colors for BB creation
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        


    @torch.no_grad()
    def detect(self, source):

        img, im0 = self.load_image(source)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3: # Single batch -> single image
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img, augment=False)[0] # We do not need any augment during inference time

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)

        # Post - Process detections
        det = pred[0] # detections per image but as we have just 1 image, it is the 0th index
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()             
            return det.detach().cpu().numpy()

        return None

    
    def load_image(self, img0):
        '''
        Load and pre process the image
        args: img0: Path of image or numpy image in 'BGR" format
        '''
        if isinstance(img0, str): img0 = cv2.imread(img0)  # BGR
        assert img0 is not None, 'Image Not Found '

        # Padded resize
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img, img0