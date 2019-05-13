#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String
import sensor_msgs.msg
import geometry_msgs.msg
import tf2_ros

from cv_bridge import CvBridge
import numpy as np

# Spartan
import spartan.utils.utils as spartanUtils
import spartan.utils.ros_utils as rosUtils
import director.transformUtils as transformUtils

class RosHeatmapVis():

    def __init__(self):
        rospy.init_node('talker', anonymous=True)
        self.pub = rospy.Publisher('chatter', String, queue_size=10)
        self.image_topics = dict()
        self.bridge = CvBridge()
        self.staticTransformBroadcaster = tf2_ros.StaticTransformBroadcaster()

    def clean_name_for_ros(self, topic_name):
        return topic_name.replace("-", "_")

    def setup_topic_if_new(self, topic_name):
        if topic_name not in self.image_topics:
            clean_topic_name = self.clean_name_for_ros(topic_name)
            self.image_topics[topic_name] = dict()

            self.image_topics[topic_name]["color_latest"] = None
            self.image_topics[topic_name]["color_pub"] = rospy.Publisher("/imgs/"+clean_topic_name+"/rgb/image_raw", sensor_msgs.msg.Image, queue_size=1)
            
            self.image_topics[topic_name]["depth_latest"] = None
            self.image_topics[topic_name]["depth_pub"] = rospy.Publisher("/imgs/"+clean_topic_name+"/depth/image_raw", sensor_msgs.msg.Image, queue_size=1)

            self.image_topics[topic_name]["cam_info_latest"] = None
            self.image_topics[topic_name]["rgb_cam_info_pub"] = rospy.Publisher("/imgs/"+clean_topic_name+"/rgb/camera_info", sensor_msgs.msg.CameraInfo, queue_size=1)
            self.image_topics[topic_name]["depth_cam_info_pub"] = rospy.Publisher("/imgs/"+clean_topic_name+"/depth/camera_info", sensor_msgs.msg.CameraInfo, queue_size=1)


    def update_rgb(self, topic_name, rgb, depth, pose, K):
        """
        topic_name: string
        rgb: cv2 img
        depth:
        pose: a dict with quaternion, translation keys
        K: 3,3 numpy matrix
        """
        self.setup_topic_if_new(topic_name)

        ros_time = rospy.Time.now()
        frame_name = topic_name + "_frame"
        self.publish_pose_transform(frame_name, pose, ros_time)

        self.image_topics[topic_name]["color_latest"] = rgb
        img_msg = self.bridge.cv2_to_imgmsg(rgb, encoding="passthrough")
        img_msg.header.stamp = ros_time
        img_msg.header.frame_id = frame_name
        self.image_topics[topic_name]["color_pub"].publish(img_msg)

        self.image_topics[topic_name]["depth_latest"] = depth

        depth = np.array(depth, dtype=np.uint16)
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="16UC1")
        depth_msg.header.stamp = ros_time
        depth_msg.header.frame_id = frame_name
        self.image_topics[topic_name]["depth_pub"].publish(depth_msg)

        cam_info_msg = sensor_msgs.msg.CameraInfo()
        cam_info_msg.header.stamp = ros_time
        cam_info_msg.header.frame_id = frame_name
        cam_info_msg.distortion_model = "plumb_bob"
        cam_info_msg.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        cam_info_msg.height = 480
        cam_info_msg.width = 640
        cam_info_msg.K = K.reshape(9).tolist()
        P = K.reshape(9).tolist()
        P.insert(3,0.0)
        P.insert(8,0.0)
        P.append(0.0)
        cam_info_msg.P = P
        cam_info_msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.image_topics[topic_name]["rgb_cam_info_pub"].publish(cam_info_msg)
        self.image_topics[topic_name]["depth_cam_info_pub"].publish(cam_info_msg)
        print "updated rgb"


        
    def publish_pose_transform(self, frame_name, pose, ros_time):
        # opticalToLinkVtk = spartanUtils.transformFromPose(pose)
        # bodyToLinkVtk = RosHeatmapVis.transformOpticalFrameToBodyFrame(opticalToLinkVtk)
        # bodyToLinkPoseDict = spartanUtils.poseFromTransform(bodyToLinkVtk)

        bodyToLink = rosUtils.ROSTransformMsgFromPose(pose)

        transform_stamped = geometry_msgs.msg.TransformStamped()
        T_world_camera_transform = bodyToLink
        transform_stamped.transform = T_world_camera_transform
        transform_stamped.child_frame_id = frame_name
        transform_stamped.header.frame_id = "base"
        transform_stamped.header.stamp = ros_time

        self.staticTransformBroadcaster.sendTransform(transform_stamped)

    @staticmethod
    def transformOpticalFrameToBodyFrame(opticalFrame):
        rpy = [-90,0,-90]
        opticalToBody = transformUtils.frameFromPositionAndRPY([0,0,0], rpy)
        bodyFrame = transformUtils.concatenateTransforms([opticalToBody.GetLinearInverse(), opticalFrame])
        return bodyFrame

    def run(self):
        rate = rospy.Rate(10) # 10hz
        while not rospy.is_shutdown():
            
            hello_str = "hello world %s" % rospy.get_time()
            rospy.loginfo(hello_str)
            self.pub.publish(hello_str)

            rate.sleep()

if __name__ == '__main__':
    hp = RosHeatmapVis()
    hp.run()