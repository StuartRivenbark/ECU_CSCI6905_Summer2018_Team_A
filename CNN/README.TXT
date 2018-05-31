This is the two run commands needed to execute this classifier.
You will need to make appropriate folder name substitutions to suit your environment:


**replace '.' with working directory**

To train model
python retrain.py
 --bottleneck_dir=".\CNN\bottlenecks"
 --how_many_training_steps 500
 --model_dir=".\CNN\models"
 --output_graph=".\CNN\outputgraph\retrained_graph.pb"
 --output_labels=".\outputlabels\retrained_labels.txt"
 --image_dir ".\CNN\chest_xray\train"

To test model against an individual image
python label_image.py
 ".\CNN\chest_xray\test\PNEUMONIA\person1_virus_6.jpg"
 ".\CNN\outputlabels\retrained_labels.txt" 
 ".\CNN\outputgraph\retrained_graph.pb"

To run full test suite
python runTests.py If you use the same file system structure, the script will find the needed parameters. If not, they can be specified or overwritten like below)

--test_path .\CNN\chest_xray\test\
--labels_path .\CNN\outputlabels\retrained_labels.txt
--graph_path .\CNN\outputgraph\retrained_graph.pb