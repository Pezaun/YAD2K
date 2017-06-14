#!/usr/bin/env python
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os
import sys


def letter_image(im, net_input_dim=(224,224,3)):
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
    list_path   = "/home/gabriel/datasets/X_Dataset_segmentation_3K/train.txt"
    images_path = "/home/gabriel/datasets/X_Dataset_segmentation_3K/images"
    labels_path = "/home/gabriel/datasets/X_Dataset_segmentation_3K/labels_voc"
    classes = {"butt":0, "breast":1, "frontalm":2, "frontalf":3}

    with open(list_path, "r") as f:
        instances_name = f.read().splitlines()

    images_data = np.zeros((len(instances_name), 416, 416, 3)).astype(np.uint8)
    labels_data = np.empty((len(instances_name))).astype(np.object)

    for i, instance in enumerate(instances_name):
        print i
        img_path = os.path.join(images_path, instance)
        img_data = letter_image(cv2.imread(img_path), (416, 416, 3))
        images_data[i] = img_data
        
        with open(os.path.join(labels_path, instance.split("/")[-1][:-4] + ".xml"), "r") as f:                
            xml_tree = ET.parse(f)
        xml_root = xml_tree.getroot()
        boxes = []
        for obj in xml_root.iter("object"):
            cls = obj.find("name").text
            if cls in ["male","female"]:
                continue
            box = obj.find("bndbox")
            boxes += [[classes[obj.find("name").text],box.find("xmin").text, box.find("ymin").text, 
                                                  box.find("xmax").text, box.find("ymax").text]]        
        labels_data[i] = np.array(boxes).astype(np.int)
    print "Saving..."

    np.savez_compressed("data.npz", images=images_data, boxes=labels_data)
