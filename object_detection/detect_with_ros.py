#!/usr/bin/env python
import os
import sys
import tarfile
import time

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as SensorImage

import tensorflow as tf
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util


class ObjectDetection:
    # PATH_TO_CKPT = '../mobilenet_v1' + '/frozen_inference_graph.pb'
    PATH_TO_CKPT = os.path.join(os.path.expanduser('~'), 'Inference', 'frozen_inference_graph.pb')
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join(os.path.expanduser('~'), 'Inference', 'label_map.pbtxt')
    NUM_CLASSES = 4

    IMAGE_PUBLISHER = '/deep_detection/object_detection'
    IMAGE_SUBSCRIBER = '/usb_cam/image_raw'

    def __init__(self):
        rospy.init_node('deep_detection')

        self.frame = None
        self.cv_bridge = CvBridge()

        self.image_publisher = rospy.Publisher(self.IMAGE_PUBLISHER,
                                               SensorImage, queue_size=100)
        self.image_subscriber = rospy.Subscriber(
            self.IMAGE_SUBSCRIBER, SensorImage, self.image_msg_callback)

        self.object_detection()

        rospy.spin()

    def load_frozen_model(self, frozen_model_path, split_model=True):
        print('>>>>>>> Loading frozen model into memory <<<<<<<<')
        if not split_model:
            print('>>>>> Not spliting model for inference <<<<<')
            detection_graph = tf.Graph()
            with detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(frozen_model_path, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
            return detection_graph, None, None
        else:
            print('>>>>> Spliting model for optimized inference <<<<<')
            # load a frozen Model and split it into GPU and CPU graphs
            # Hardcoded for ssd_mobilenet

            ###################################################################
            input_graph = tf.Graph()
            with tf.Session(graph=input_graph):
                if ssd_shape == 600:
                    shape = 7326
                else:
                    shape = 1917
                score = tf.placeholder(tf.float32, shape=(None, shape, num_classes),
                                       name="Postprocessor/convert_scores")
                expand = tf.placeholder(tf.float32, shape=(None, shape, 1, 4),
                                        name="Postprocessor/ExpandDims_1")
                for node in input_graph.as_graph_def().node:
                    if node.name == "Postprocessor/convert_scores":
                        score_def = node
                    if node.name == "Postprocessor/ExpandDims_1":
                        expand_def = node
            #####################################################################
            #####################################################################
            detection_graph = tf.Graph()
            with detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(frozen_model_path, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    dest_nodes = ['Postprocessor/convert_scores', 'Postprocessor/ExpandDims_1']

                    edges = {}
                    name_to_node_map = {}
                    node_seq = {}
                    seq = 0
                    for node in od_graph_def.node:
                        n = _node_name(node.name)
                        name_to_node_map[n] = node
                        edges[n] = [_node_name(x) for x in node.input]
                        node_seq[n] = seq
                        seq += 1
                    for d in dest_nodes:
                        assert d in name_to_node_map, "%s is not in graph" % d

                    nodes_to_keep = set()
                    next_to_visit = dest_nodes[:]

                    while next_to_visit:
                        n = next_to_visit[0]
                        del next_to_visit[0]
                        if n in nodes_to_keep:
                            continue
                        nodes_to_keep.add(n)
                        next_to_visit += edges[n]

                    nodes_to_keep_list = sorted(list(nodes_to_keep), key=lambda n: node_seq[n])
                    nodes_to_remove = set()

                    for n in node_seq:
                        if n in nodes_to_keep_list:
                            continue
                        nodes_to_remove.add(n)
                    nodes_to_remove_list = sorted(list(nodes_to_remove), key=lambda n: node_seq[n])

                    keep = graph_pb2.GraphDef()
                    for n in nodes_to_keep_list:
                        keep.node.extend([copy.deepcopy(name_to_node_map[n])])

                    remove = graph_pb2.GraphDef()
                    remove.node.extend([score_def])
                    remove.node.extend([expand_def])
                    for n in nodes_to_remove_list:
                        remove.node.extend([copy.deepcopy(name_to_node_map[n])])

                    with tf.device('/gpu:0'):
                        tf.import_graph_def(keep, name='')
                    with tf.device('/cpu:0'):
                        tf.import_graph_def(remove, name='')
            #####################################################################
            return detection_graph, score, expand

    def load_labelmap():
        print('>>>>>>> Loading labelmap from label_map.pbtxt <<<<<<<<')
        label_map = label_map_util.load_labelmap(label_path)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=num_classes, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index

    def detection(detection_graph, category_index, score, expand):
        print(">>>>> Building Graph fpr object detection <<<<<")
        # Session Config: allow seperate GPU/CPU adressing and limit memory allocation
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device)
        config.gpu_options.allow_growth = allow_memory_growth
        cur_frames = 0
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph, config=config) as sess:
                # Define Input and Ouput tensors
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                if split_model:
                    score_out = detection_graph.get_tensor_by_name('Postprocessor/convert_scores:0')
                    expand_out = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
                    score_in = detection_graph.get_tensor_by_name(
                        'Postprocessor/convert_scores_1:0')
                    expand_in = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')
                    # Threading
                    gpu_worker = SessionWorker("GPU", detection_graph, config)
                    cpu_worker = SessionWorker("CPU", detection_graph, config)
                    gpu_opts = [score_out, expand_out]
                    cpu_opts = [detection_boxes, detection_scores,
                                detection_classes, num_detections]
                    gpu_counter = 0
                    cpu_counter = 0
                # Start Video Stream and FPS calculation
                fps = FPS2(fps_interval).start()
                video_stream = WebcamVideoStream(video_input, width, height).start()
                cur_frames = 0
                print("> Press 'q' to Exit")
                print('> Starting Detection')
                while video_stream.isActive():
                    # actual Detection
                    if split_model:
                        # split model in seperate gpu and cpu session threads
                        if gpu_worker.is_sess_empty():
                            # read video frame, expand dimensions and convert to rgb
                            image = video_stream.read()
                            image_expanded = np.expand_dims(
                                cv2.cvtColor(image, cv2.COLOR_BGR2RGB), axis=0)
                            # put new queue
                            gpu_feeds = {image_tensor: image_expanded}
                            if visualize:
                                gpu_extras = image  # for visualization frame
                            else:
                                gpu_extras = None
                            gpu_worker.put_sess_queue(gpu_opts, gpu_feeds, gpu_extras)

                        g = gpu_worker.get_result_queue()
                        if g is None:
                            # gpu thread has no output queue. ok skip, let's check cpu thread.
                            gpu_counter += 1
                        else:
                            # gpu thread has output queue.
                            gpu_counter = 0
                            score, expand, image = g["results"][0], g["results"][1], g["extras"]

                            if cpu_worker.is_sess_empty():
                                # When cpu thread has no next queue, put new queue.
                                # else, drop gpu queue.
                                cpu_feeds = {score_in: score, expand_in: expand}
                                cpu_extras = image
                                cpu_worker.put_sess_queue(cpu_opts, cpu_feeds, cpu_extras)

                        c = cpu_worker.get_result_queue()
                        if c is None:
                            # cpu thread has no output queue. ok, nothing to do. continue
                            cpu_counter += 1
                            time.sleep(0.005)
                            continue  # If CPU RESULT has not been set yet, no fps update
                        else:
                            cpu_counter = 0
                            boxes, scores, classes, num, image = c["results"][0], c[
                                "results"][1], c["results"][2], c["results"][3], c["extras"]
                    else:
                        # default session
                        image = video_stream.read()
                        image_expanded = np.expand_dims(
                            cv2.cvtColor(image, cv2.COLOR_BGR2RGB), axis=0)
                        boxes, scores, classes, num = sess.run(
                            [detection_boxes, detection_scores, detection_classes, num_detections],
                            feed_dict={image_tensor: image_expanded})

                    # Visualization of the results of a detection.
                    if visualize:
                        vis_util.visualize_boxes_and_labels_on_image_array(
                            image,
                            np.squeeze(boxes),
                            np.squeeze(classes).astype(np.int32),
                            np.squeeze(scores),
                            category_index,
                            use_normalized_coordinates=True,
                            line_thickness=8)
                        if vis_text:
                            cv2.putText(image, "fps: {}".format(fps.fps_local()), (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
                        cv2.imshow('object_detection', image)
                        # Exit Option
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        cur_frames += 1
                        # Exit after max frames if no visualization
                        for box, score, _class in zip(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)):
                            if cur_frames % det_interval == 0 and score > det_th:
                                label = category_index[_class]['name']
                                print("> label: {}\nscore: {}\nbox: {}".format(label, score, box))
                        if cur_frames >= max_frames:
                            break
                    fps.update()

        # End everything
        if split_model:
            gpu_worker.stop()
            cpu_worker.stop()
        fps.stop()
        video_stream.stop()
        cv2.destroyAllWindows()
        print('> [INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
        print('> [INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    def image_msg_callback(self, img):
        self.frame = self.cv_bridge.imgmsg_to_cv2(img)
        pass

    def object_detection(self):
        sys.path.append("..")

        # tar_file = tarfile.open('ssd_mobilenet_v1_coco_11_06_2017.tar.gz')
        # for files in tar_file.getmembers():
        #     file_name = os.path.basename(files.name)
        #     if 'frozen_inference_graph.pb' in file_name:
        #         tar_file.extract(files, os.getcwd())

        print(' ========= Loading detection graph from checkpoint ===================')
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                Writer = tf.summary.FileWriter("./logs/graph", sess.graph)
                print(' =================== Fetching input tensor and output tensors  ===================')
                while not rospy.is_shutdown():
                    image_np = self.frame

                    if image_np is not None:
                        print('--------------- processing ..... -------------------')
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

                        image_message = self.cv_bridge.cv2_to_imgmsg(image_np, encoding='rgb8')
                        if image_message != None:
                            print('[INFO]: Sending image message through image publisher')
                        self.image_publisher.publish(image_message)
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            cv2.destroyAllWindows()
                            break
                    else:
                        print('[ERROR]:There is no image feeded throught the network')


if __name__ == '__main__':
    ObjectDetection()
