#!/usr/bin/env python
import pickle
import numpy as np
import random
from collections import defaultdict
import xml.etree.ElementTree as ET
import os
import PIL

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
        self.find_max_boxes()
        print "Max objcts:", self.max_objs
        print "Done!"

    def data_generator(self, batch, shuffle=False):
        X = np.zeros((batch, net_input_dim[0], net_input_dim[1], net_input_dim[2])).astype(np.float32)
        Y = np.zeros((batch, self.max_objs, 5)).astype(np.float32)

        batch_index = 0
        instances_index = 0
        while True:
            im_data = [PIL.Image.open(self.images_paths[instances_index])
            im_key = self.images_paths[instances_index].split("/")[-1][:-4]

            im_data = im_data.resize((416, 416), PIL.Image.BICUBIC)            
            im_data = np.array(im_data, dtype=np.float)
            im_data /= 255.

            # im_data = self.letter_image(im_data, net_input_dim).astype(np.float32)
            # im_data /= 255.0
            # im_data = im_data[...,::-1]

            batch_index += 1
            batch_index = batch_index % batch
            instances_index += 1

            X[batch_index] = im_data
            Y[batch_index] = self.features_dict[im_key]

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
                xml_tree = ET.parse(f)
                xml_root = xml_tree.getroot()
                obj_ct = 0
                for obj in xml_root.iter("object"):
                    cls = obj.find("name").text
                    if cls in ["male","female"]:
                        continue
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

if __name__ == "__main__":
    list_path   = "/home/gabriel/datasets/X_Dataset_segmentation_3K/valid.txt"
    images_path = "/home/gabriel/datasets/X_Dataset_segmentation_3K/images"
    labels_path = "/home/gabriel/datasets/X_Dataset_segmentation_3K/labels_voc"
    loader = XDatasetForYOLOLoader(images_path, labels_path, list_path)
