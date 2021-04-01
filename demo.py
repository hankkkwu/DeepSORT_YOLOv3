#! /usr/bin/env python
# -*- coding: utf-8 -*-

# import warnings
import cv2
import numpy as np
from PIL import Image
import copy
from yolo_nms import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection as ddet
from tools import generate_detections as gdet
from keras import backend

backend.clear_session()
# warnings.filterwarnings('ignore')

# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

yolo = YOLO()

#Definition of the parameters
max_cosine_distance = 0.5 # cosine distance threshold
nn_budget = None
nms_max_overlap = 0.3 # NMS overlap threshold

#deep_sort
#model_filename = 'model_data/market1501.pb'
model_filename = 'model_data/mars.pb'
encoder = gdet.create_box_encoder(model_filename,batch_size=1)

metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)


def main(input_image):
    image = copy.deepcopy(input_image)
    image_h, image_w, _ = image.shape
    bbox_thick = int(0.6 * (image_h + image_w) / 600)

    boxs, class_names = yolo.inference(image)
    features = encoder(image, boxs)

    # score to 1.0 here
    # detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
    detections = [Detection(bbox, class_name, 1.0, feature) for bbox, class_name, feature in zip(boxs, class_names, features)]

    # Run non-maxima suppression.
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
    filtered_detections = [detections[i] for i in indices]

    # Call the tracker
    tracker.predict()
    tracker.update(filtered_detections)

    # draw bounding boxes after non_max_suppression
    # for det in filtered_detections:
    #     bbox = det.to_tlbr()
    #     class_name = det.class_name
    #     cv2.rectangle(image,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 1)
    #     cv2.putText(image, str(class_name),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (255,255,255), 2)

    for track in tracker.tracks:
        # track include [x, P, track_id, hits, n_init, age, max_age, time_since_update, state, feature]
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        bbox = track.to_tlbr()
        color = [int(c) for c in COLORS[track.track_id % len(COLORS)]]

        # draw bounding box
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, bbox_thick)

        # draw track id
        t_size = cv2.getTextSize(str(track.track_id), 0, 0.5, thickness=bbox_thick//2)[0]
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0]) + t_size[0], int(bbox[1]) - t_size[1] - 3), color, -1)
        cv2.putText(image, str(track.track_id), (int(bbox[0]), int(bbox[1]-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), bbox_thick//2, lineType=cv2.LINE_AA)

    return image



"""
# 6 - Process Video
"""
# Commented out IPython magic to ensure Python compatibility.
from moviepy.editor import VideoFileClip

video_file = "./test_video/MOT16-13-raw.mp4" #25 FPS
# video_file = "./test_video/Taiwan_Highway_8.MOV"

clip = VideoFileClip(video_file)
white_clip = clip.fl_image(main)
white_clip.write_videofile("./test_video/MOT16-13-raw_out_2.mp4",audio=False)
