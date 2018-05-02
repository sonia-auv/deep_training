from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import imghdr
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.saved_model.python.saved_model import reader
import tensorflow.contrib.tensorrt as trt


def get_frozen_graph(graph_file):
"""Read Frozen Graph file from disk."""
with tf.gfile.FastGFile(graph_file, "rb") as f:
graph_def = tf.GraphDef()
graph_def.ParseFromString(f.read())
return graph_def



def write_graph_to_file(graph_name, graph_def, output_dir):
"""Write Frozen Graph file to disk."""
output_path = os.path.join(output_dir, graph_name)
with tf.gfile.GFile(output_path, "wb") as f:
f.write(graph_def.SerializeToString())


def get_trt_graph(graph_name, graph_def, precision_mode, output_dir,
output_node, batch_size=24, workspace_size=10<<30):
trt_graph = trt.create_inference_graph(
graph_def, [output_node], max_batch_size=batch_size,
max_workspace_size_bytes=workspace_size,
precision_mode=precision_mode)
write_graph_to_file(graph_name, trt_graph, output_dir)



if __name__ == "__main__":
frozen_graph_def = get_frozen_graph('/home/spark/Models/frozen/frozen/mobilenet_v1/frozen_inference_graph.pb')

get_trt_graph('test',frozen_graph_def , "FP32", '/home/spark/',
'SemanticPredictions')