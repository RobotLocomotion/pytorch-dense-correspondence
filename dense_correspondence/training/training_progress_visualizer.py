import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import dense_correspondence_manipulation.utils.utils as utils
from dense_correspondence.evaluation.plotting import normalize_descriptor

class TrainingProgressVisualizer():

	def __init__(self):
		self.title = "Training Progress"
		self.init_keypoint_data()
		self.init_plot()
		# load the keypoint data

	def make_ticklabels_invisible(self,fig,iteration):
	    for i, ax in enumerate(fig.axes):
	        #ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
	        ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, grid_alpha=0, bottom=False, top=False, left=False, right=False)
	    ax_label = plt.subplot(self.gs1[-1,0])
	    ax_label.clear()
	    ax_label.text(0.5, 0.5, "iteration: %d" % (iteration), va="center", ha="center")
	    ax_label.axis('off')
		

	def init_plot(self):
		self.fig = plt.figure()
		plt.axis('off')
		#self.fig.suptitle("Descriptor learning over time")
		#self.fig.set_figheight(5)
		#self.fig.set_figwidth(10)
		self.gs1 = GridSpec(3, 3)
		self.gs1.update(left=0.05, right=0.48, wspace=0.05)
		self.ax1 = plt.subplot(self.gs1[:-1, :])
		
		# self.ax2 = plt.subplot(self.gs1[-1, :-1])
		# self.ax3 = plt.subplot(self.gs1[-1, -1])

		print len(self.keypoint_data), "is number of images"
		print np.sqrt(len(self.keypoint_data)), "is sqrt"
		self.num_square_cells = int(np.ceil(np.sqrt(len(self.keypoint_data))))
		print self.num_square_cells, "is size of gridspec"

		self.gs2 = GridSpec(self.num_square_cells*2, self.num_square_cells)
		self.gs2.update(left=0.55, right=0.98, hspace=0.05)
		# ax4 = plt.subplot(gs2[:, :-1])
		# ax5 = plt.subplot(gs2[:-1, -1])
		# ax6 = plt.subplot(gs2[-1, -1])
		# self.ax4 = plt.subplot(self.gs2[0,0])
		# self.ax4.scatter([0,1,2],[3,5,5])
		# self.ax5 = plt.subplot(self.gs2[1,0])
		# self.ax6 = plt.subplot(self.gs2[0,1])
		# self.ax7 = plt.subplot(self.gs2[1,1])
		# self.ax8 = plt.subplot(self.gs2[1,-1])
		# self.ax9 = plt.subplot(self.gs2[-1,-1])
		#self.make_ticklabels_invisible(self.fig)


	def init_keypoint_data(self):
		keypoint_data_path = "/home/peteflo/code/data_volume/pdc/evaluation_labeled_data/shoe_annotated_keypoints.yaml"
		#keypoint_data_path = "/home/peteflo/code/data_volume/pdc/evaluation_labeled_data/single_scene_red_nike_keypoints.yaml"
		#keypoint_data_path = "/home/peteflo/code/data_volume/pdc/evaluation_labeled_data/multiple_scenes_red_nike_keypoints.yaml"
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
			image_a_rgb, _, mask, _ = dataset.get_rgbd_mask_pose(scene_name, image_a_idx)

			image_a_tensor = dataset.rgb_image_to_tensor(image_a_rgb)

			res = network.forward_single_image_tensor(image_a_tensor).data.cpu().numpy()
			res[:,:,0] = res[:,:,0] * np.asarray(mask)
			res[:,:,1] = res[:,:,1] * np.asarray(mask)
			
			res_numpy = normalize_descriptor(res)
			res_numpy = np.clip(res_numpy, a_min = 0.0, a_max = 1.0)
			res_numpy = 255 * res_numpy
			res_numpy = res_numpy.astype(np.uint8)
			res_numpy[:,:,0] = res_numpy[:,:,0] * np.asarray(mask)
			res_numpy[:,:,1] = res_numpy[:,:,1] * np.asarray(mask)

			#plt.figure(1+index)

			# to handle D=2
			print res_numpy.shape
			
			res_numpy_img = np.zeros((res_numpy.shape[0], res_numpy.shape[1], 3)).astype(np.uint8)
			res_numpy_img[:,:,0] = res_numpy[:,:,0]
			res_numpy_img[:,:,1] = res_numpy[:,:,1]
			res_numpy_img[:,:,2] = res_numpy[:,:,1]
			print res_numpy_img.shape


			index_row = (index / self.num_square_cells) * 2
			index_col = index % self.num_square_cells

			ax_rgb = plt.subplot(self.gs2[index_row,index_col])
			ax_res = plt.subplot(self.gs2[index_row+1,index_col])

			ax_rgb.clear()
			ax_rgb.imshow(image_a_rgb)

			ax_res.clear()
			ax_res.imshow(res_numpy_img)

			for i in img["image"]["pixels"]:
				keypoint = i["keypoint"]
				print int(i["u"])
				print int(i["v"])
				print res[int(i["v"]), int(i["u"])]
				i["descriptor"]  = res[int(i["v"]), int(i["u"])]
				ax_rgb.scatter(i["u"], i["v"], marker=self.keypoint_to_marker(keypoint), s=10, color=self.object_id_to_color(object_id))
				#ax_res.scatter(i["u"], i["v"], marker=self.keypoint_to_marker(keypoint), color=self.object_id_to_color(object_id))


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
				

	def update(self, dataset, network, iteration, now_training_object_id=None):

		#plt.clf()
		#plt.cla()

		# get first image
		# forward it through network
		# get keypoint descriptors out of network
		self.identify_descriptors(dataset, network)

		self.ax1.clear()
		for index, img in enumerate(self.keypoint_data):
			object_id = img["image"]["object_id"]
			color = self.object_id_to_color(object_id)
			for i in img["image"]["pixels"]:
				keypoint = i["keypoint"]
				descriptor = i["descriptor"]
				marker = self.keypoint_to_marker(keypoint)
				self.ax1.scatter(descriptor[0], descriptor[1], alpha=0.5, color=color, marker=marker, label=object_id+keypoint)
		
		#l1 = plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
		#plt.tight_layout()

		plt.rcParams.update({'font.size': 6})
		self.make_ticklabels_invisible(self.fig, iteration)
		if object_id is not None:
			ax_label = plt.subplot(self.gs1[-1,-1])
			ax_label.clear()
			ax_label.text(0.5, 0.5, "training: %s" % (now_training_object_id), va="center", ha="center")
			ax_label.axis('off')
		self.ax1.set_title('Descriptors (D=2) of keypoints')
		self.fig.savefig("./progress_vis_2/new_"+str(iteration).zfill(5)+".png", dpi=300)

		#plt.draw()
		#plt.pause(0.001)
		#plt.show()
		print "updated"
