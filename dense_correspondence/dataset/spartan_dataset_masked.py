from dense_correspondence_dataset_masked import DenseCorrespondenceDataset

import os
import glob

class SpartanDataset(DenseCorrespondenceDataset):
    def __init__(self, debug=False):
    	self.logs_root_path = self.load_from_config_yaml("relative_path_to_spartan_logs")


        # use all scenes
        self.scenes = os.listdir(self.logs_root_path)
        
        blacklist = ["14_background"]
        for blacklisted_scene in blacklist:
            self.scenes.remove(blacklisted_scene)

        self.init_length()
        print "Using SpartanDataset with:"
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