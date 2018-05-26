1) You will need to substitute your root directory with mine which is:
   C:\Users\Stuart\Documents\Python Scripts\ECU_CSCI6905_Summer2018_Team_A\CNN

2) To run the pasta training:
   Step a) cd C:\Users\Stuart\Documents\Python Scripts\ECU_CSCI6905_Summer2018_Team_A\CNN
   Step b) python retrain.py --bottleneck_dir="C:\Users\Stuart\Documents\Python Scripts\ECU_CSCI6905_Summer2018_Team_A\CNN\bottlenecks" --how_many_training_steps 500 --model_dir="C:\Users\Stuart\Documents\Python Scripts\ECU_CSCI6905_Summer2018_Team_A\CNN\models" --output_graph="C:\Users\Stuart\Documents\Python Scripts\ECU_CSCI6905_Summer2018_Team_A\CNN\outputgraph\retrained_graph.pb" --output_labels="C:\Users\Stuart\Documents\Python Scripts\ECU_CSCI6905_Summer2018_Team_A\CNN\outputlabels\retrained_labels.txt" --image_dir "C:\Users\Stuart\Documents\Python Scripts\ECU_CSCI6905_Summer2018_Team_A\CNN\chest_xray\train"
3) To see how the training did:
   python label_image.py "C:\Users\Stuart\Documents\Python Scripts\ECU_CSCI6905_Summer2018_Team_A\CNN\test\RandomXRay.jpg" "C:\Users\Stuart\Documents\Python Scripts\ECU_CSCI6905_Summer2018_Team_A\CNN\outputlabels\retrained_labels.txt"  "C:\Users\Stuart\Documents\Python Scripts\ECU_CSCI6905_Summer2018_Team_A\CNN\outputgraph\retrained_graph.pb"