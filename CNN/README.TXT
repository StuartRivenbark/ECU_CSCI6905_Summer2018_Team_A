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