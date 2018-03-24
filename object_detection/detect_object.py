#!/usr/bin/env python
import numpy as np
import os
import six.moves.urllib as urllib
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as SensorImage
import sys
import tensorflow as tf
import tarfile
import requests
import cv2
import rospy

#cap = cv2.VideoCapture(0)
rospy.init_node('deep_detection')

sys.path.append("..")

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = 'ssd_mobilenet_v1_coco_11_06_2017' + '/frozen_inference_graph.pb'

image_cv = CvBridge()


def image_msg_callback(img):
    image = image_cv.imgmsg_to_cv2(img)
    pass


image_publisher = rospy.Publisher('/deep_detection/object_detection', SensorImage, queue_size=100)
image_subscriber = rospy.Subscriber('/usb_cam/image_raw', SensorImage, image_msg_callback)


# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

if not os.path.exist(MODEL_FILE):
    url = DOWNLOAD_BASE + MODEL_FILE
    filename = url.split("/")[-1]
    with open(filename, "wb") as file_:
        r = requests.get(url)
        file_.write(r.content)


tar_file = tarfile.open('ssd_mobilenet_v1_coco_11_06_2017.tar.gz')
for files in tar_file.getmembers():
    file_name = os.path.basename(files.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(files, os.getcwd())

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[8]:

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        Writer = tf.summary.FileWriter("./logs/graph", sess.graph)
        while not rospy.is_shutdown():
            image_np = cap.read()
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            image_message = image_cv.cv2_to_imgmsg(image_np, encoding="rgb8")
            image_publisher.publish(image_message)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
