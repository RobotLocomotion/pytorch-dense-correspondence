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

	def identify_descriptors(self, dataset, network):
		for index, img in enumerate(self.keypoint_data):
			print "img"
			scene_name = img["image"]["scene_name"]
			image_a_idx = img["image"]["image_idx"]

			# can simplify to just get rgb
			image_a_rgb, _, _, _ = dataset.get_rgbd_mask_pose(scene_name, image_a_idx)

			image_a_tensor = dataset.rgb_image_to_tensor(image_a_rgb)

			res = network.forward_single_image_tensor(image_a_tensor).data.cpu().numpy()
			
			res_numpy = normalize_descriptor(res)
			res_numpy = np.clip(res_numpy, a_min = 0.0, a_max = 1.0)
			res_numpy = 255 * res_numpy
			res_numpy = res_numpy.astype(np.uint8)

			plt.figure(1+index)
			plt.imshow(res_numpy)

			for i in img["image"]["pixels"]:
				print i["keypoint"]
				print int(i["u"])
				print int(i["v"])
				print res[int(i["v"]), int(i["u"])]
				plt.scatter(i["u"], i["v"])
				


	def update(self, dataset, network):
		for i in range(len(self.keypoint_data)+1):
			plt.figure(i)
			plt.clf()
		
		# get first image
		# forward it through network
		# get keypoint descriptors out of network
		self.identify_descriptors(dataset, network)

		plt.figure(0)
		plt.scatter([1,2,4],[1,3,5], alpha=0.5, label="brown_boot")
		plt.scatter(0,2, color="red", marker="X", alpha=0.5, label="brown_boot")
		plt.scatter(4,3, alpha=0.5, label="brown_boot")
		plt.scatter(1,1, alpha=0.5, label="brown_boot")
		plt.legend()

		plt.draw()
		plt.pause(0.001)
		plt.show()
		print "updated"
