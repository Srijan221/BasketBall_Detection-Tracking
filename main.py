from detection_algorithm import *
from tracking_algorithms import *
from  YoloDS_binder import *
from PIL import Image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/best_yolov7-tiny.pt', help='model.pt path(s)')
    parser.add_argument('--input', type=str, default='./IO_data/input/video/45248-45436.avi', help='input video path')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='./IO_data/output/video/45248-45436.avi', help='output video path')  # file/folder, 0 for webcam
    parser.add_argument('--show_live', action='store_true', default=False, help='to display video')
    parser.add_argument('--count_objects', action = 'store_true', default=False, help='To count objects tracked')
    parser.add_argument('--verbose', default=2, help='verbose 0,1,2')
    parser.add_argument('--graph', default = "3D", help='show graph 2D,3D')
    parser.add_argument('--draw_line', action = 'store_true', default=False, help='To draw live trajectory')
    opt = parser.parse_args()
    print(opt)

    detector = Detector(classes = [0]) # it'll detect ONLY Basketball
    detector.load_model(opt.weights,) # pass the path to the trained weight file        
    tracker = BasketBallTracker(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector)
    # output = None will not save the output video
    tracker.track_video(opt.input, output=opt.output, show_live = opt.show_live, count_objects = opt.count_objects, verbose = opt.verbose, graph = opt.graph, draw_line = opt.draw_line)