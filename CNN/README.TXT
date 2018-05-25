This is the two run commands needed to execute this classifer.
You will need to make appropriate folder name substitutions to suit your environment:

python retrain.py
 --bottleneck_dir="C:\Users\RivenbarkSW\IdeaProjects\ECU_CSCI6905_Summer2018_Team_A\CNN\bottlenecks"
 --how_many_training_steps 500
 --model_dir="C:\Users\RivenbarkSW\IdeaProjects\ECU_CSCI6905_Summer2018_Team_A\CNN\models"
 --output_graph="C:\Users\RivenbarkSW\IdeaProjects\ECU_CSCI6905_Summer2018_Team_A\CNN\outputgraph\retrained_graph.pb"
 --output_labels="C:\Users\RivenbarkSW\IdeaProjects\ECU_CSCI6905_Summer2018_Team_A\CNN\outputlabels\retrained_labels.txt"
 --image_dir "C:\Users\RivenbarkSW\IdeaProjects\ECU_CSCI6905_Summer2018_Team_A\CNN\images"

python label_image.py
 "C:\Users\RivenbarkSW\IdeaProjects\ECU_CSCI6905_Summer2018_Team_A\CNN\MDBlueCrab.jpg"
 "C:\Users\RivenbarkSW\IdeaProjects\ECU_CSCI6905_Summer2018_Team_A\CNN\outputlabels\retrained_labels.txt" 
 "C:\Users\RivenbarkSW\IdeaProjects\ECU_CSCI6905_Summer2018_Team_A\CNN\outputgraph\retrained_graph.pb"
