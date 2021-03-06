import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IECore

def non_max_suppression_fast(boxes, nmsThreshold):
    # if there are no boxes, return an empty list
    boxes = np.array(boxes)
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]+boxes[:,0]
    y2 = boxes[:,3]+boxes[:,1]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > nmsThreshold)[0])))
    # return only the bounding boxes that were picked using the
    # integer data
    return boxes[pick].astype("int").tolist(), np.array(pick)

class YOLO():
    def __init__(self):
        """
        - YOLO takes an image as input. We should set the dimension of the image to a fixed number.
        - The default choice is often 416x416.
        - YOLO applies thresholding and non maxima suppression, define a value for both
        - Load the classes, model configuration (cfg file) and pretrained weights (weights file) into variables
        - If the image is 416x416, the weights must be corresponding to that image
        - Load the network with OpenCV.dnn function
        """
        self.confThreshold = 0.5
        self.nmsThreshold = 0.55
        self.inpWidth = 608
        self.inpHeight = 608
        classesFile = "./model_data/coco.names"
        self.classes = None
        with open(classesFile,'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        modelConfiguration = "./model_data/yolov3.cfg";
        modelWeights = "./model_data/yolov3.weights";
        self.net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        #self.net = cv2.dnn.readNetFromModelOptimizer('./model_data/frozen_darknet_yolov3_model.xml',
        #                                             './model_data/frozen_darknet_yolov3_model.bin')
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        #self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        #self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

    def getOutputsNames(self):
        '''
        Get the names of the output layers
        '''
        # Get the names of all the layers in the network
        layersNames = self.net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]


    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        '''
        Draw a bounding box around a detected object given the box coordinates
        Later, we could repurpose that to display an ID
        '''
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), thickness=5)
        label = '%.2f' % conf
        # Get the label for the class name and its confidence
        if self.classes:
            assert(classId < len(self.classes))
            label = '%s:%s' % (self.classes[classId], label)

        #Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness=3)
        return frame

    def postprocess(self,frame, outs):
        """
        Postprocessing step. Take the output out of the neural network and interpret it.
        We should use that output to apply NMS thresholding and confidence thresholding
        We should use the output to draw the bounding boxes using the dramPred function
        """
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        classIds = []
        confidences = []
        boxes = []
        class_names = []
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        #print("outs:", len(outs))
        for out in outs:
            #print("out:", len(out))
            #for detection in out[0]:
            for detection in out:
                #print("detection:", len(detection))
                scores = detection[5:]
                #print("score length:", scores[0])
                classId = np.argmax(scores)
                #print("id:", classId)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    if self.classes:
                        assert(classId < len(self.classes))
                        class_name = self.classes[classId]
                        class_names.append(class_name)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
                    # boxes.append([left, top, left+width, top+height])
        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        #indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold) # Previous Indices
        # filtered_boxes = []
        # if len(boxes)>0:
        #     filtered_boxes, indices = non_max_suppression_fast(boxes, self.nmsThreshold) # CALLING THE OTHER NMS FORMULA
            # for idx, box in enumerate(filtered_boxes):
            #     i = indices[idx]
            #     left, top, width, height = box
            #     output_image = self.drawPred(frame,classIds[i], confidences[i], left, top, left + width, top + height)

        return boxes, class_names

    def inference(self,image):
        """
        Main loop.
        Input: Image
        Output: Frame with the drawn bounding boxes
        """
        #image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(image, 1/255, (self.inpWidth, self.inpHeight), [0,0,0], 1, crop=False)
        # Sets the input to the network
        self.net.setInput(blob)
        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.getOutputsNames())
        #outs = self.net.forward()
        # Remove the bounding boxes with low confidence
        # final_frame, boxes = self.postprocess(image, outs)
        boxes, class_names = self.postprocess(image, outs)
        return boxes, class_names
