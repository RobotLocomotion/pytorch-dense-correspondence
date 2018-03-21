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

    def get_pose(self, time_filename):
        time = self.get_time(time_filename)
        scene_directory = time_filename.split("images")[0]
        pose_list = self.get_pose_list(scene_directory)
        pose_labelfusion = self.get_pose_from_list(time, pose_list)
        pose_matrix4 = self.labelfusion_pose_to_homogeneous_transform(pose_labelfusion)
        return pose_matrix4

    def get_time(self, time_filename):
        with open (time_filename) as f:
            content = f.readlines()
        return int(content[0])/1e6

    def get_pose_list(self, scene_directory):
        posegraph_filename = os.path.join(scene_directory, "posegraph.posegraph")
        with open(posegraph_filename) as f:
            content = f.readlines()
        pose_list = [x.strip().split() for x in content]
        return pose_list

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

    def labelfusion_pose_to_homogeneous_transform(self, lf_pose):
        homogeneous_transform = self.quaternion_matrix([lf_pose[6], lf_pose[3], lf_pose[4], lf_pose[5]])
        homogeneous_transform[0,3] = lf_pose[0]
        homogeneous_transform[1,3] = lf_pose[1]
        homogeneous_transform[2,3] = lf_pose[2]
        return homogeneous_transform