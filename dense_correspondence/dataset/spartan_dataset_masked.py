from dense_correspondence_dataset_masked import DenseCorrespondenceDataset

import os
import glob

class SpartanDataset(DenseCorrespondenceDataset):
    def __init__(self, debug=False, mode="train"):
    	self.logs_root_path = self.load_from_config_yaml("relative_path_to_spartan_logs")

        train_test_config = os.path.join(os.path.dirname(__file__), "train_test_config", "0001_drill_test.yaml")
        self.set_train_test_split_from_yaml(train_test_config)
        
        if mode == "test":
            self.set_test_mode()
        
        self.init_length()
        print "Using SpartanDataset:"
        print "   - in", self.mode, "mode"
        print "   - number of scenes:", len(self.scenes)
        print "   - total images:    ", self.num_images_total

        DenseCorrespondenceDataset.__init__(self, debug=debug)

    def get_pose(self, rgb_filename):
        scene_directory = rgb_filename.split("images")[0]
        index = self.get_index(rgb_filename)
        pose_list = self.get_pose_list(scene_directory, "images.posegraph")
        pose_elasticfusion = self.get_pose_from_list(int(index), pose_list)
        pose_matrix4 = self.elasticfusion_pose_to_homogeneous_transform(pose_elasticfusion)
        return pose_matrix4

    def get_pose_from_list(self, index, pose_list):
        pose = pose_list[index]
        pose = [float(x) for x in pose[1:]]
        return pose

    def get_index(self, rgb_filename):
        prefix = rgb_filename.split("_rgb")[0]
        return prefix.split("images/")[1]

    def get_mask_filename(self, rgb_filename):
        images_masks_dir = os.path.join(os.path.dirname(os.path.dirname(rgb_filename)), "image_masks")
        index = self.get_index(rgb_filename)
        mask_filename = images_masks_dir+"/"+index+"_mask.png"
        return mask_filename