import os, sys, argparse, subprocess
logfile = open('CNN_TestResults.txt', 'a+')

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

def log(msg):
	print(msg)
	sys.stdout.flush()
	logfile.write(str(msg) + '\n')
	logfile.flush()

def run(args):
	log("Path to Test Directory: " + args.test_path)
	log("Path to Label File: " + args.labels_path)
	log("Path to Graph File: " + args.graph_path)
	for dir in os.listdir(args.test_path):
		log("directory: " + dir)
		if dir == "NORMAL":
			log("RUNNING TESTS FOR NORMAL XRAYS")
			current_dir = args.test_path + "\\NORMAL"
			index = 1
			for file in os.listdir(current_dir):	
				exec_path = os.path.join(current_dir, file)
				log("TEST #:" + str(index))
				log("IMAGE:" + exec_path)
				#log("python label_image.py " + exec_path + " " + args.labels_path + " " + args.graph_path)
				#subprocess.call(['python', 'label_image.py', exec_path, args.labels_path, args.graph_path], shell=True)
				run = subprocess.Popen(['python', 'label_image.py', exec_path, args.labels_path, args.graph_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
				for line in run.stdout:
					log(line.decode('utf-8'))
				run.wait()
				log("-----------------------------------------------------------++")
				index = index + 1
		elif dir == "PNEUMONIA":
			log("RUNNING TESTS FOR PNEUMONIA XRAYS")
			current_dir = args.test_path + "\\PNEUMONIA"
			index = 1
			for file in os.listdir(current_dir):
				exec_path = os.path.join(current_dir, file)
				log("TEST #:" + str(index))
				log("IMAGE:" + exec_path)
				#subprocess.call(["python label_image.py " + args.test_path + " " + args.labels_path + " " + args.graph_path], shell=True)
				run = subprocess.Popen(['python', 'label_image.py', exec_path, args.labels_path, args.graph_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
				for line in run.stdout:
					log(line.decode('utf-8'))
				run.wait()
				log("-----------------------------------------------------------++")
				index = index + 1
		else:
			print("SKIPPING DIRECTORY: " + dir)

if __name__ == '__main__':
	default_dir_path = os.path.dirname(os.path.realpath(__file__))
	log("default_dir_path: " + default_dir_path)
	default_test_path = os.path.join(default_dir_path, 'chest_xray\\test')
	default_label_path = os.path.join(default_dir_path, 'outputlabels\\retrained_labels.txt')
	default_graph_path = os.path.join(default_dir_path, 'outputgraph\\retrained_graph.pb')
	
	parser = argparse.ArgumentParser(description='Test wrapper')
	parser.add_argument('--test_path', help='Specify the directory containing test data', default=default_test_path)
	parser.add_argument('--labels_path', help='Specify the file containing label data', default=default_label_path)
	parser.add_argument('--graph_path', help='Specify the file containg graph data', default=default_graph_path)
	args = parser.parse_args()
	run(args)