import numpy as np
import matplotlib.pyplot as plt
import dense_correspondence_manipulation.utils.utils as utils
from dense_correspondence.evaluation.plotting import normalize_descriptor

class TrainingProgressVisualizer():

	def __init__(self):
		self.title = "Training Progress"
		self.init_plot()
		self.init_keypoint_data()
		# load the keypoint data

	def init_plot(self):
		plt.figure(0)
		plt.scatter(0,2, alpha=0.5, label="brown_boot")
		plt.scatter(1,1, alpha=0.5, label="brown_boot")
		plt.ion()
		plt.legend()
		plt.show()

	def init_keypoint_data(self):
		keypoint_data_path = "/home/peteflo/code/data_volume/pdc/evaluation_labeled_data/shoe_annotated_keypoints.yaml"
		self.keypoint_data = utils.getDictFromYamlFilename(keypoint_data_path)
		print "KEYPOINT DATA"
		print self.keypoint_data
		self.check_keypoint_data()
		self.init_descriptors_in_keypoint_dict()

	def check_keypoint_data(self):
		"""
		This function verifies that nothing is fishy with the labeled keypoint data.
		"""
		keypoints = []
		img = self.keypoint_data[0]
		for i in img["image"]["pixels"]:
				print i["keypoint"]
				keypoints.append(i["keypoint"])

		for index, img in enumerate(self.keypoint_data[1:]):
			for i, v in enumerate(img["image"]["pixels"]):
				if v["keypoint"] not in keypoints:
					print "MISMATCH IN TRAINING DATA"
					quit()
				if keypoints[i] != v["keypoint"]:
					print "MISMATCH IN TRAINING DATA"
					quit()

	def init_descriptors_in_keypoint_dict(self):
		for index, img in enumerate(self.keypoint_data):
			for i, v in enumerate(img["image"]["pixels"]):
				v["descriptor"] = [0.0, 0.0, 0.0]
	
	def identify_descriptors(self, dataset, network):
		for index, img in enumerate(self.keypoint_data):
			print "img"
			scene_name = img["image"]["scene_name"]
			image_a_idx = img["image"]["image_idx"]
			object_id = img["image"]["object_id"]

			# can simplify to just get rgb
			image_a_rgb, _, _, _ = dataset.get_rgbd_mask_pose(scene_name, image_a_idx)

			image_a_tensor = dataset.rgb_image_to_tensor(image_a_rgb)

			res = network.forward_single_image_tensor(image_a_tensor).data.cpu().numpy()
			
			res_numpy = normalize_descriptor(res)
			res_numpy = np.clip(res_numpy, a_min = 0.0, a_max = 1.0)
			res_numpy = 255 * res_numpy
			res_numpy = res_numpy.astype(np.uint8)

			plt.figure(1+index)

			# to handle D=2
			print res_numpy.shape
			
			res_numpy_img = np.zeros((res_numpy.shape[0], res_numpy.shape[1], 3)).astype(np.uint8)
			res_numpy_img[:,:,0] = res_numpy[:,:,0]
			res_numpy_img[:,:,1] = res_numpy[:,:,1]
			res_numpy_img[:,:,2] = res_numpy[:,:,1]
			print res_numpy_img.shape

			plt.imshow(res_numpy_img)

			for i in img["image"]["pixels"]:
				keypoint = i["keypoint"]
				print int(i["u"])
				print int(i["v"])
				print res[int(i["v"]), int(i["u"])]
				i["descriptor"]  = res[int(i["v"]), int(i["u"])]
				plt.scatter(i["u"], i["v"], marker=self.keypoint_to_marker(keypoint), color=self.object_id_to_color(object_id))


	def keypoint_to_marker(self, keypoint):
		if keypoint == "toe":
			return "o"
		if keypoint == "top_of_shoelaces":
			return "x"
		if keypoint == "heel":
			return "<"


	def object_id_to_color(self, object_id):
		if object_id == "shoe_green_nike":
			return "green"
		if object_id == "shoe_gray_nike":
			return "gray"
		if object_id == "shoe_red_nike":
			return "red"
		if object_id == "shoe_brown_boot":
			return "brown"
				

	def update(self, dataset, network):
		for i in range(len(self.keypoint_data)+1):
			plt.figure(i)
			plt.clf()
		
		# get first image
		# forward it through network
		# get keypoint descriptors out of network
		self.identify_descriptors(dataset, network)

		plt.figure(0)

		for index, img in enumerate(self.keypoint_data):
			object_id = img["image"]["object_id"]
			color = self.object_id_to_color(object_id)
			for i in img["image"]["pixels"]:
				keypoint = i["keypoint"]
				descriptor = i["descriptor"]
				marker = self.keypoint_to_marker(keypoint)
				plt.scatter(descriptor[0], descriptor[1], alpha=0.5, color=color, marker=marker, label=object_id+keypoint)
		
		l1 = plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
		plt.tight_layout()

		plt.draw()
		plt.pause(0.001)
		plt.show()
		print "updated"
