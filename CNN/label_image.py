import os, sys

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(len(sys.argv))

# change this as you see fit
image_path = sys.argv[1]
labels_path = "retrained_labels.txt"
graph_path = "retrained_graph.pb"
print("Arg Length:")
print("==========")
print(sys.argv)
print("===========")
if len(sys.argv) > 2:
    labels_path = sys.argv[2]
if len(sys.argv) > 3:
    graph_path = sys.argv[3]

print("Using the following arguments:")
print("=============================")
print("image  path = {0}".format(image_path))
print("labels path = {0}".format(labels_path))
print("graph  path = {0}".format(graph_path))
print("=========================================================================================================")

# Read in the image_data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile(labels_path)]

# Unpersists graph from file
with tf.gfile.FastGFile(graph_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})
    
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))