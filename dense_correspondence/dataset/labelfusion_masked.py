from dense_correspondence_dataset_masked import DenseCorrespondenceDataset

import os

class LabelFusionDataset(DenseCorrespondenceDataset):
    def __init__(self, debug):
    	self.logs_root_path = self.load_from_config_yaml("relative_path_to_labelfusion_logs_test")

        # 5 drill scenes
        self.scenes = ["2017-06-13-12",
                       "2017-06-13-01",
                       "2017-06-13-15",
                       "2017-06-13-16",
                       "2017-06-13-20"]

        #self.scenes = ["2017-06-13-12"] # just drill scene in tool area

        self.init_length()
        print "Using LabelFusionDataset with:"
        print "   - number of scenes:", len(self.scenes)
        print "   - total images:    ", self.num_images_total

        DenseCorrespondenceDataset.__init__(self, debug=debug)

    def get_pose(self, rgb_filename):
    	time_filename = self.get_time_filename(rgb_filename)
        time = self.get_time(time_filename)
        scene_directory = time_filename.split("images")[0]
        pose_list = self.get_pose_list(scene_directory, "posegraph.posegraph")
        pose_elasticfusion = self.get_pose_from_list(time, pose_list)
        pose_matrix4 = self.elasticfusion_pose_to_homogeneous_transform(pose_elasticfusion)
        return pose_matrix4

    def get_time_filename(self, rgb_image):
        prefix = rgb_image.split("rgb")[0]
        time_filename = prefix+"utime.txt"
        return time_filename

    def get_mask_filename(self, rgb_image):
        prefix = rgb_image.split("rgb")[0]
        mask_filename = prefix+"labels.png"
        return mask_filename

    def get_time(self, time_filename):
        with open (time_filename) as f:
            content = f.readlines()
        return int(content[0])/1e6

    def get_pose_from_list(self, time, pose_list):
        if (time <= float(pose_list[0][0])):
            pose = pose_list[0]
            pose = [float(x) for x in pose[1:]]
            return pose
        for pose in pose_list:
            if (time <= float(pose[0])):
                pose = [float(x) for x in pose[1:]]
                return pose
        print "did not find matching pose, must be at end of list"
        return pose