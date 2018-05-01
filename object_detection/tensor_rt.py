from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import IPython
import os

from tensorflow.python.platform import gfile


os.environ["CUDA_VISIBLE_DEVICES"]="0"

#INPUT_TENSOR_NAME = 'image_tensor:0'

output_node_name  = [
    'detection_boxes:0',
    'detection_scores:0',
    'detection_classes:0',
    'num_detections:0',
]

batch_size = 24
workspace_size = 4000000000
precision='FP32' #“FP32”, “FP16” or “INT8”

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.67)

file_name = '/home/spark/Downloads/'

with gfile.FastGFile(file_name,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.reset_default_graph()
    frozen_graph_def=tf.Graph()

    IPython.embed()


    trt_graph = trt.create_inference_graph(
        input_graph_def = frozen_graph_def,
        outputs = output_node_name,
        max_batch_size=batch_size,
        max_workspace_size_bytes=workspace_size,
        precision_mode=precision,
        minimum_segment_size=3)

