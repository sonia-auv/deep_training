#!/usr/bin/env python
import numpy as np
import os
import time
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as SensorImage
import sys
import tensorflow as tf
import tarfile
import cv2
import rospy


class ObjectDetection:

    def __init__(self):
        rospy.init_node('deep_detection')

        self.frame = None
        self.cv_bridge = CvBridge()

        self.image_publisher = rospy.Publisher('/deep_detection/object_detection', SensorImage, queue_size=100)
        self.image_subscriber = rospy.Subscriber('/provider_vision/Front_GigE', SensorImage, self.image_msg_callback)

        self.object_detection()

        rospy.spin()

    def image_msg_callback(self, img):
        self.frame = self.cv_bridge.imgmsg_to_cv2(img)
        pass

    def object_detection(self):
        sys.path.append("..")

        from object_detection.utils import label_map_util

        from object_detection.utils import visualization_utils as vis_util

        PATH_TO_CKPT = 'ssd_mobilenet_v1_coco_11_06_2017' + '/frozen_inference_graph.pb'

        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

        NUM_CLASSES = 90

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
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                Writer = tf.summary.FileWriter("./logs/graph", sess.graph)
                while not rospy.is_shutdown():
                    image_np = self.frame
                    if image_np is not None:
                        start_time = time.time()
                        image_np.setflags(write=1)
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
                        (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
                                                                            feed_dict={image_tensor: image_np_expanded})
                        # Visualization of the results of a detection.
                        vis_util.visualize_boxes_and_labels_on_image_array(
                            image_np,
                            np.squeeze(boxes),
                            np.squeeze(classes).astype(np.int32),
                            np.squeeze(scores),
                            category_index,
                            use_normalized_coordinates=True,
                            line_thickness=8)
                        print "FPS: ", 1.0 / float(time.time() - start_time)
                        image_message = self.cv_bridge.cv2_to_imgmsg(image_np, encoding="rgb8")
                        self.image_publisher.publish(image_message)
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            cv2.destroyAllWindows()
                            break

if __name__ == '__main__':
    ObjectDetection()