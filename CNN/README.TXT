This is the two run commands needed to execute this classifer.
You will need to make appropriate folder name substitutions to suit your environment:

python image_classifier.py
 --bottleneck_dir="C:\Users\RivenbarkSW\Documents\Python Scripts\TensorFlow\CNNProject\bottlenecks"
 --how_many_training_steps 500
 --model_dir="C:\Users\RivenbarkSW\Documents\Python Scripts\TensorFlow\CNNProject\models"
 --output_graph="C:\Users\RivenbarkSW\Documents\Python Scripts\TensorFlow\CNNProject\outputgraph\retrained_graph.pb"
 --output_labels="C:\Users\RivenbarkSW\Documents\Python Scripts\TensorFlow\CNNProject\outputlabels\retrained_labels.txt"
 --image_dir "C:\Users\RivenbarkSW\Documents\Python Scripts\TensorFlow\CNNProject\images"

python label_image.py "C:\Users\RivenbarkSW\Documents\Python Scripts\TensorFlow\CNNProject\MDBlueCrab.jpg"


PLEASE IGNORE MY NOTES BELOW. It's just so I remember this stuff when I get home from work.
=============================================================================================
python image_classifier.py --bottleneck_dir="C:\Users\RivenbarkSW\Documents\Python Scripts\TensorFlow\CNNProject\bottlenecks" --how_many_training_steps 500 --model_dir="C:\Users\RivenbarkSW\Documents\Python Scripts\TensorFlow\CNNProject\models" --output_graph="C:\Users\RivenbarkSW\Documents\Python Scripts\TensorFlow\CNNProject\outputgraph\retrained_graph.pb" --output_labels="C:\Users\RivenbarkSW\Documents\Python Scripts\TensorFlow\CNNProject\outputlabels\retrained_labels.txt" --image_dir "C:\Users\RivenbarkSW\Documents\Python Scripts\TensorFlow\CNNProject\images"

python image_classifier.py --bottleneck_dir="C:\Users\RivenbarkSW\Documents\Python Scripts\TensorFlow\CNNProject\bottlenecks" --how_many_training_steps 500 --model_dir="C:\Users\RivenbarkSW\Documents\Python Scripts\TensorFlow\CNNProject\models" --output_graph="C:\Users\RivenbarkSW\Documents\Python Scripts\TensorFlow\CNNProject\outputgraph\retrained_graph.pb" --output_labels="C:\Users\RivenbarkSW\Documents\Python Scripts\TensorFlow\CNNProject\outputlabels\retrained_labels.txt" --image_dir "C:\Users\RivenbarkSW\Documents\Python Scripts\TensorFlow\CNNProject\images" --input_layer=input --output_layer=MobilenetV1/Predictions/Reshape_1 --input_mean=-127 --input_std=127

--input_layer=input --output_layer=MobilenetV1/Predictions/Reshape_1 --graph=/tmp/mobilenet_v1_224/frozen_graph.pb --logtostderr --input_mean=-127 --input_std=127 --image=third_party/tensorflow/examples/label_image/data/grace_hopper.jpg --input_width=224 --input_height=224 --labels=/tmp/mobilenet_v1_224/labels.txt

python label_image.py "C:\Users\RivenbarkSW\Documents\Python Scripts\TensorFlow\CNNProject\MDBlueCrab.jpg"

python label_image.py "C:\Users\RivenbarkSW\Documents\Python Scripts\TensorFlow\CNNProject\MDBlueCrab.jpg" --output_layer=final_result --labels="C:\Users\RivenbarkSW\Documents\Python Scripts\TensorFlow\CNNProject\outputlabels\retrained_labels.txt" --image="C:\Users\RivenbarkSW\Documents\Python Scripts\TensorFlow\CNNProject\MDBlueCrab.jpg" --graph="C:\Users\RivenbarkSW\Documents\Python Scripts\TensorFlow\CNNProject\outputgraph\retrained_graph.pb" --input_mean=-127 --input_std=127 --input_width=224 --input_height=224

https://github.com/tensorflow/tensorflow/issues/11250
https://github.com/tensorflow/hub/blob/master/docs/tutorials/image_retraining.md

python label_image.py --image="C:\Users\RivenbarkSW\Documents\Python Scripts\TensorFlow\CNNProject\MDBlueCrab.jpg" --graph="C:\Users\RivenbarkSW\Documents\Python Scripts\TensorFlow\CNNProject\outputgraph\retrained_graph.pb" --labels="C:\Users\RivenbarkSW\Documents\Python Scripts\TensorFlow\CNNProject\outputlabels\retrained_labels.txt"

usage: label_image.py [-h] [--image IMAGE] [--graph GRAPH] [--labels LABELS]
                      [--input_height INPUT_HEIGHT]
                      [--input_width INPUT_WIDTH] [--input_mean INPUT_MEAN]
                      [--input_std INPUT_STD] [--input_layer INPUT_LAYER]
                      [--output_layer OUTPUT_LAYER]

tensorboard --logdir="C:\Users\RivenbarkSW\Documents\Python Scripts\TensorFlow\CNNProject\logs"


tf.trainable_variables():
=========================
[<tf.Variable 'final_retrain_ops/weights/final_weights:0' shape=(2048, 5) dtype=float32_ref>, <tf.Variable 'final_retrain_ops/biases/final_biases:0' shape=(5,) dtype=float32_ref>]

