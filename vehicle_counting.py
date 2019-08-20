#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 27th January 2018
#----------------------------------------------

# Imports
import tensorflow as tf
import argparse

# Object detection imports
from utils import backbone
from api import object_counting_api

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
	help="path to input video")
ap.add_argument("-r", "--roi", type=str, required=True,
	help="ROI Line")
args = vars(ap.parse_args())

input_video = args["input"]

# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
detection_graph, category_index = backbone.set_model('ssd_mobilenet_v2_coco_2018_03_29', 'mscoco_label_map.pbtxt')

is_color_recognition_enabled = 1 # set it to 1 for enabling the color prediction for the detected objects
roi = args["roi"] # roi line position
deviation = 2 # the constant that represents the object counting area

object_counting_api.cumulative_object_counting_y_axis(input_video, detection_graph, category_index, is_color_recognition_enabled, roi, deviation) # counting all the objects
