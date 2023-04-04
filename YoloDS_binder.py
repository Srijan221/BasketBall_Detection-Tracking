'''
This file binds the YoloV7 code with DeepSort algorithm
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # comment out below line to enable tensorflow logging outputs
import time
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from scipy.signal import savgol_filter

import cv2
import matplotlib.pyplot as plt

from tensorflow.compat.v1 import ConfigProto # DeepSORT implementation uses tf1.x

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

# import binders
from tracking_algorithms import create_box_encoder
from detection_algorithm import *

#for plotting graph
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

from tkinter import *

 # load configuration for object detector
config = ConfigProto()
config.gpu_options.allow_growth = True



class BasketBallTracker:
    '''
    Class to Wrap ANY detector  of YOLO type with DeepSORT
    '''
    def __init__(self, reID_model_path:str, detector, max_cosine_distance:float=0.4, nn_budget:float=None, nms_max_overlap:float=1.0,  ):
        '''
        args: 
            reID_model_path: Path of the model which uses generates the embeddings for the cropped area for Re identification
            detector: object of YOLO models or any model which gives you detections as [x1,y1,x2,y2,scores, class]
            max_cosine_distance: Cosine Distance threshold for "SAME" person matching
            nn_budget:  If not None, fix samples per class to at most this number. Removes the oldest samples when the budget is reached.
            nms_max_overlap: Maximum NMs allowed for the tracker
        '''
        self.detector = detector
        self.nms_max_overlap = nms_max_overlap
        self.class_names = {0: 'Basketball'}

        # initialize deep sort
        self.encoder = create_box_encoder(reID_model_path, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget) # calculate cosine distance metric
        self.tracker = Tracker(metric) # initialize tracker

    def smooth_trajectory(self,points):
        '''
        To visualise the path of the Basketball real-time
        '''

        split_trajectory = [ [points[0]] ]
        current_split = 0
        for i in range(1, len(points)):
            distance = np.linalg.norm(np.array(points[i - 1])-np.array(points[i]))
            if distance < 15:
                split_trajectory[current_split].append(points[i])
            else:
                split_trajectory.append([ points[i] ])
                current_split += 1
        
        new_X, new_Y = [], []

        for split in split_trajectory:
            X = np.array([ p[0] for p in split ])
            Y = np.array([ p[1] for p in split ])

            if len(Y) > 20:
                y_hat = savgol_filter(Y, 15, 3)
                Y = [ int(round(y)) for y in y_hat ]

            new_X.extend(X)
            new_Y.extend(Y)

        return list(zip(new_X, new_Y))

    
    def draw_trajectory(self,image, points, max_distance):
        for i in range(1, len(points)):
            thickness = 2
            distance = np.linalg.norm(np.array(points[i - 1])-np.array(points[i]))
            if distance < max_distance:
                image = cv2.line(image, points[i - 1], points[i], (255, 0, 255), thickness)

        return image
    
    def track_video(self,video:str, output:str, show_live:bool=False, count_objects:bool=False, verbose:int = 0, graph:str = None, draw_line:bool = True):
        '''
        Track any given webcam or video
        args: 
            video: path to input video or set to 0 for webcam
            output: path to output video
            show_live: Whether to show live video tracking. Press the key 'q' to quit
            count_objects: count objects being tracked on screen
            verbose: print details on the screen allowed values 0,1,2
            graph: plot and save graph of the (x_center, y_center) coordinates vs time, allowed values 2D,3D
            draw_line: to visualise the trajectory of ball on video allowed true or false
        '''
        try: # begin video capture
            vid = cv2.VideoCapture(int(video))
        except:
            vid = cv2.VideoCapture(video)

        out = None
        if output: # get video ready to save locally if flag is set
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))  # by default VideoCapture returns float instead of int
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(output, codec, fps, (width, height))

        x_coord = []
        y_coord = []
        time_frames = []
        frame_num = 0
        points = []
        while True: # while video is running
            return_value, frame = vid.read()
            if not return_value:
                print('Video has ended')
                break
            frame_num +=1

            if verbose >= 1:start_time = time.time()

            # -----------------------------------------DETECTION CODE STARTS HERE -----------------------------------------------------------------
            
            yolo_dets = self.detector.detect(frame.copy())  # To Get the detections
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if yolo_dets is None:
                bboxes = []
                scores = []
                classes = []
                num_objects = 0
            
            else:
                bboxes = yolo_dets[:,:4]
                bboxes[:,2] = bboxes[:,2] - bboxes[:,0] # convert from xyxy to xywh
                bboxes[:,3] = bboxes[:,3] - bboxes[:,1]

                scores = yolo_dets[:,4]
                classes = yolo_dets[:,-1]
            # ---------------------------------------- DETECTION PART COMPLETED ---------------------------------------------------------------------
            
            names = ["Basketball"]
            names = np.array(names)
            count = len(names)

            if count_objects:
                cv2.putText(frame, "Objects detected: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 0), 2)

            # ---------------------------------- DeepSORT TRACKING CODE STARTS HERE work ------------------------------------------------------------
            
            
            features = self.encoder(frame, bboxes) # encode detections and feed to tracker. [No of BB / detections per frame, embed_size]
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)] # [No of BB per frame] deep_sort.detection.Detection object

            cmap = plt.get_cmap('tab20b') #initialize color map
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            boxs = np.array([d.tlwh for d in detections])  # run non-maxima supression below
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]       

            self.tracker.predict()  # Call the tracker
            self.tracker.update(detections) #  update using Kalman Gain

            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()
        
                color = colors[int(track.track_id) % len(colors)]  # draw bbox on screen
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + " : " + str(track.track_id),(int(bbox[0]), int(bbox[1]-11)),0, 0.6, (255,255,255),1, lineType=cv2.LINE_AA)    

                if verbose == 2:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                
                if graph:
                    x1,y1,x2,y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                    x_cen = (x1 + x2) // 2
                    y_cen = (y1 + y2) // 2
                    cv2.circle(frame, (x_cen,y_cen),
                        radius=1,
                        color=(255, 0, 0),
                        thickness=3)
                    
                    x_coord.append(x_cen)
                    y_coord.append(y_cen)
                    print(y_cen)
                    time_frames.append(frame_num)
                
                if draw_line:
                    new_point = x_cen,y_cen
                    points.append(new_point)
                    points = list(filter(lambda pt: pt is not None, points))

                    if len(points) > 30:
                        points = self.smooth_trajectory(points)

                    frame  = self.draw_trajectory(frame, points, 30)
                    
                    
                    
            # -------------------------------- Tracker work ENDS here -----------------------------------------------------------------------
            if verbose >= 1:
                fps = 1.0 / (time.time() - start_time) # calculate frames per second of running detections
                if not count_objects: print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)}")
                else: print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)} || Objects tracked: {count}")
            
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if output: out.write(result) # save output video

            if show_live:
                cv2.imshow("Output Video", result)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        cv2.destroyAllWindows()

        if graph == "3D":
                ax = plt.axes(projection='3d')
                ax.plot3D(x_coord,time_frames, y_coord, 'red')
                ax.set_xlabel('X')
                ax.set_ylabel('Time')
                ax.invert_zaxis()
                ax.set_zlabel('Y')
                plt.ylim(ymin=0.0)
                graph_path = output.split("/")[-1]
                plt.savefig(f"IO_data/output/{graph_path}_3D_graph.png")
                plt.show()
        elif graph == "2D":
            graph, (plot1, plot2) = plt.subplots(1, 2)
            plot1.plot(x_coord,time_frames)
            plot1.set_title("X-Values vs Time")
            plot1.set_xlabel('X-Coordinates')
            plot1.set_ylabel('Time Frames')

            plot2.plot(time_frames, y_coord)
            plot2.invert_yaxis()
            plot2.set_xlabel('Time Frames')
            plot2.set_ylabel('Y-Coordinates')

            plot2.set_title("Time vs Y-Values")
            graph_path = output.split("/")[-1]
            plt.tight_layout()
            plt.savefig(f"IO_data/output/{graph_path}_2D_graph.png")
            plt.show()