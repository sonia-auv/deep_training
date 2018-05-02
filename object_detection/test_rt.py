import os
import tensorflow as tf
from tensorflow.contrib import tensorrt as trt

# Path to frozen Object Detection graph
network_dir = "/home/spark/Models/frozen/frozen/mobilenet_v1/frozen_inference_graph.pb"

print("Loading graph")
detection_graph = tf.Graph()
with detection_graph.as_default():
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(network_dir, "rb") as fid:
        serialized_graph = fid.read()
        graph_def.ParseFromString(serialized_graph)
        _ = tf.import_graph_def(graph_def, name='')
print("Finished loading graph")

trt_graph = trt.create_inference_graph(
    input_graph_def=graph_def,
    outputs=[
        "detection_boxes",
        "detection_classes",
        "detection_scores",
        "num_detections"
    ],
    max_batch_size=1,
    max_workspace_size_bytes=1 << 32,
    precision_mode="FP16",
    minimum_segment_size=2
)
tf.reset_default_graph()
tf.import_graph_def(graph_def=trt_graph)