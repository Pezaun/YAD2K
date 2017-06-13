#!/usr/bin/env python
import cv2
import pickle
import numpy as np
import random
from collections import defaultdict
import xml.etree.ElementTree as ET
import os
import sys
from yad2k.models.keras_yolo import (preprocess_true_boxes, yolo_body,
                                     yolo_eval, yolo_head, yolo_loss)

class XDatasetForYOLOLoader:
    def __init__(self, IMAGES_PATH, LABELS_PATH, SET_PATH, DIM=(416,416,3)):
        print "Creating Loader...",
        with open(SET_PATH, "r") as f:
            self.images_paths = f.read().splitlines()
        self.instance_count = len(self.images_paths)
        self.dim = DIM
        self.labels_path = LABELS_PATH
        self.max_objs = 0
        print "Find objects count..."
        self.labels_dic = defaultdict()
        self.classes = {"butt":0, "breast":1, "frontalm":2, "frontalf":3}
        self.find_max_boxes()        
        print "Max objcts:", self.max_objs
        print "Done!"

    def data_generator(self, batch, shuffle=False):
        X = np.zeros((batch, self.dim[0], self.dim[1], self.dim[2])).astype(np.float32)
        Y = np.zeros((batch, self.max_objs, 5)).astype(np.int)

        batch_index = 0
        instances_index = 0
        while True:
            im_key = self.images_paths[instances_index].split("/")[-1][:-4]
            im_data = cv2.imread(self.images_paths[instances_index])
            orig_size = np.array([im_data.shape[1], im_data.shape[0]])
            orig_size = np.expand_dims(orig_size, axis=0)
                        
            im_data = self.letter_image(im_data, self.dim).astype(np.float32)
            im_data /= 255.0

            batch_index += 1
            batch_index = batch_index % batch
            instances_index += 1

            X[batch_index] = im_data

            boxes = np.asarray(self.labels_dic[im_key]).astype(np.float)

            boxes_xy = 0.5 * (boxes[:, 3:5] + boxes[:, 1:3])
            boxes_wh = boxes[:, 3:5] - boxes[:, 1:3]
            boxes_xy = boxes_xy / orig_size
            boxes_wh = boxes_wh / orig_size
            boxes = np.concatenate((boxes_xy, boxes_wh, boxes[:, 0:1]), axis=1)

            boxes = np.vstack((boxes, np.zeros((self.max_objs - boxes.shape[0], 5))))

            Y[batch_index] = boxes

            if instances_index == len(self.images_paths):
                instances_index = 0
                if shuffle:
                    random.shuffle(self.images_paths)

            if batch_index == 0:
                yield X, Y

    def find_max_boxes(self):
        label_files = os.listdir(self.labels_path)
        for label_file in label_files:
            with open(os.path.join(self.labels_path, label_file), "r") as f:                
                self.labels_dic[label_file[:-4]] = []
                xml_tree = ET.parse(f)
                xml_root = xml_tree.getroot()
                obj_ct = 0
                for obj in xml_root.iter("object"):
                    cls = obj.find("name").text
                    if cls in ["male","female"]:
                        continue
                    box = obj.find("bndbox")
                    self.labels_dic[label_file[:-4]] += [[self.classes[obj.find("name").text],box.find("xmin").text, box.find("ymin").text, 
                                                          box.find("xmax").text, box.find("ymax").text]]
                    obj_ct += 1
                if obj_ct > self.max_objs:
                    self.max_objs = obj_ct


    def letter_image(self, im, net_input_dim=(224,224,3)):
        w = net_input_dim[1]
        h = net_input_dim[0]
        im_out = np.ones(net_input_dim) * 127

        new_w = im.shape[1];
        new_h = im.shape[0];
        if w/float(im.shape[1]) < h/float(im.shape[0]):
            new_w = w;
            new_h = (im.shape[0] * w)/im.shape[1];
        else:
            new_h = h;
            new_w = (im.shape[1] * h)/im.shape[0];

        im_out[(h - new_h) / 2:new_h + (h - new_h) / 2,(w - new_w) / 2:new_w + (w - new_w) / 2,:] = cv2.resize(im, (new_w, new_h))
        return im_out


def get_detector_mask(boxes, anchors):
    '''
    Precompute detectors_mask and matching_true_boxes for training.
    Detectors mask is 1 for each spatial position in the final conv layer and
    anchor that should be active for the given boxes and 0 otherwise.
    Matching true boxes gives the regression targets for the ground truth box
    that caused a detector to be active or 0 otherwise.
    '''
    detectors_mask = [0 for i in range(len(boxes))]
    matching_true_boxes = [0 for i in range(len(boxes))]
    print boxes.shape
    for i, box in enumerate(boxes):
        print box.shape
        detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, [416, 416])

    return np.array(detectors_mask), np.array(matching_true_boxes)

if __name__ == "__main__":
    list_path   = "/home/gabriel/datasets/X_Dataset_segmentation_3K/valid.txt"
    images_path = "/home/gabriel/datasets/X_Dataset_segmentation_3K/images"
    labels_path = "/home/gabriel/datasets/X_Dataset_segmentation_3K/labels_voc"

    YOLO_ANCHORS = np.array(((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
                            (7.88282, 3.52778), (9.77052, 9.16828)))

    loader = XDatasetForYOLOLoader(images_path, labels_path, list_path)
    gen = loader.data_generator(2)
    for i in range(2):
        print "next..."
        data = gen.next()
        boxes = data[1]
        print boxes.shape
        print get_detector_mask(boxes, YOLO_ANCHORS)[1].max()
        # for i in range(len(boxes)):
        #     get_detector_mask(boxes[i], YOLO_ANCHORS)
        # print i, data[0].shape, data[1].shape, data[0].mean(), data[1].mean()
        # cv2.imwrite("im" + str(i) + ".jpg", (data[0][0]*255.0).astype(np.uint8))
