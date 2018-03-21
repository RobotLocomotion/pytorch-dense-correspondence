from dense_correspondence_dataset_masked import DenseCorrespondenceDataset

import os
import glob

class SpartanDataset(DenseCorrespondenceDataset):
    def __init__(self, debug):
    	self.logs_root_path = self.load_from_config_yaml("relative_path_to_spartan_logs")

        # use all scenes
        self.scenes = glob.glob(self.logs_root_path)

        self.scenes = ["processed_04_drill_long_downsampled"] # just this one scene

        self.init_length()
        print "Using SpartanDataset with:"
        print "   - number of scenes:", len(self.scenes)
        print "   - total images:    ", self.num_images_total

        DenseCorrespondenceDataset.__init__(self, debug=debug)


    def get_pose(self, rgb_filename):
        scene_directory = rgb_filename.split("images")[0]

        prefix = rgb_filename.split("_rgb")[0]
        index = prefix.split("images/")[1]
        
        pose_list = self.get_pose_list(scene_directory)
        pose_labelfusion = self.get_pose_from_list(int(index), pose_list)
        pose_matrix4 = self.labelfusion_pose_to_homogeneous_transform(pose_labelfusion)
        return pose_matrix4

    def get_time(self, index):
        return index

    def get_pose_list(self, scene_directory):
        posegraph_filename = os.path.join(scene_directory, "images.posegraph")
        with open(posegraph_filename) as f:
            content = f.readlines()
        pose_list = [x.strip().split() for x in content]
        return pose_list

    def get_pose_from_list(self, index, pose_list):
        return pose_list[index]

    def labelfusion_pose_to_homogeneous_transform(self, lf_pose):
        homogeneous_transform = self.quaternion_matrix([lf_pose[6], lf_pose[3], lf_pose[4], lf_pose[5]])
        homogeneous_transform[0,3] = lf_pose[0]
        homogeneous_transform[1,3] = lf_pose[1]
        homogeneous_transform[2,3] = lf_pose[2]
        return homogeneous_transform

    def get_mask_filename(self, rgb_filename):
        prefix = rgb_filename.split("_rgb")[0]
        index = prefix.split("images/")[1]
        images_masks_dir = os.path.join(os.path.dirname(os.path.dirname(rgb_filename)), "image_masks")
        mask_filename = images_masks_dir+"/mask_"+index+".png"
        print mask_filename
        return mask_filename