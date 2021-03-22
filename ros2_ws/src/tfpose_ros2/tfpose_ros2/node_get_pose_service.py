import time

import rclpy
from rclpy.node import Node

import cv2
import numpy as np

from cv_bridge import CvBridge

from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

from pose_interface.msg import BodyPartElm
from pose_interface.srv import GetPose

from tfpose_ros2 import common
from tfpose_ros2.estimator import TfPoseEstimator
from tfpose_ros2.networks import get_graph_path, model_wh


class GetPoseService(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.image = None
        self.image_taken = False
        self.cloud = None
        self.cloud_taken = False
        self.image_w = 432
        self.image_h = 368
        self.cloud_w = 640
        self.cloud_h = 480
        self.br = CvBridge()
        self.e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(self.image_w, self.image_h))
        self.image_subscription = self.create_subscription(
            Image,
            'camera/color/image_raw',
            self.image_listener_callback,
            10)
        self.point_cloud_subscription = self.create_subscription(
            PointCloud2,
            '/camera/pointcloud',
            self.point_cloud_listener_callback,
            10)
        self.srv = self.create_service(GetPose, 'get_pose_estimation', self.get_pose_service_callback)

        self.srv
        self.image_subscription  # prevent unused variable warning
        self.point_cloud_subscription

    def get_pose_service_callback(self, request, response):
        while (not self.image_taken and not self.cloud_taken):
            continue
        pc = pc2.read_points(self.cloud, field_names=("x", "y", "z"), skip_nans=True)
        self.get_logger().info("PointCloud read success")
        t = time.time()
        humans = self.e.inference(self.image, resize_to_default=(self.image_w > 0 and self.image_h > 0), upsample_size=4.0)
        elapsed = time.time() - t
        self.get_logger().info('inference image: in %.4f seconds.' % elapsed)
        image_out = TfPoseEstimator.draw_humans(self.image, humans, imgcopy=False)
        cv2.imwrite('/home/vlad/img.png', image_out)

        # self.get_logger().info(list(pc))
        parts = []
        if len(humans) > 0:
            for part in humans[0].body_parts.values():
                body_part_elem = BodyPartElm()
                body_part_elem.part_id = part.part_idx
                body_part_elem.x = part.x
                body_part_elem.y = part.y
                body_part_elem.confidence = part.score
                parts.append(body_part_elem)
                self.get_logger().info("PARTNAME: %s  X: %s  Y: %s" % (part.get_part_name(), part.x, part.y))
        response.person.body_part = parts
        return response

    def image_listener_callback(self, msg):
        self.image_taken = True
        self.image = self.br.imgmsg_to_cv2(msg)

    def point_cloud_listener_callback(self, msg):
        self.cloud_taken = True
        self.cloud = msg
        pass


def main(args=None):
    rclpy.init(args=args)

    get_pose_service = GetPoseService()

    rclpy.spin(get_pose_service)

    get_pose_service.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

