#!/usr/bin/env python3
from fsd_path_planning.utils.math_utils import unit_2d_vector_from_angle, rotate
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning import PathPlanner, MissionTypes, ConeTypes
import rospy
import numpy as np
import timeit
from std_msgs.msg import String, Int16, Float32
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import Odometry, Path
from fs_msgs.msg import PlannedPath
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Pose
import tf
from tf.transformations import quaternion_from_euler
import threading
# Rest of the code

planner = PathPlanner(MissionTypes.trackdrive)


class PathPlanner:
    def _init_(self):
        rospy.init_node("path_planner", anonymous=True)
        rospy.Subscriber("/slam/output/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/slam/output/markers_map", MarkerArray, self.marker_callback)
        rospy.Subscriber( "/navigation/speed_profiler/path", PlannedPath, self.curr_path_callback)
        # self.pub = rospy.Publisher('/kthfs/result', Float32, queue_size=10)

        def wrapper():
            self.convert_path(None)
            
        self.original_path_pub = rospy.Publisher("/original_path", Path, queue_size=10)
        self.generated_path_pub = rospy.Publisher("/generated_pub", Path, queue_size=10)
        rospy.Timer(rospy.Duration(1/2), self.convert_path)
        # rospy.Timer(rospy.Duration(1/2), self.generate_new_path)
        self.br = tf.TransformBroadcaster()

        self.curr_path = None
        
        self.stamp = None
        self.cones_right_raw = np.array([])
        self.cones_left_raw = np.array([])
        print("running")
        # rospy.sleep(2)
        # self.time_check()

    def generate_new_path(self):
        if self.car_x is not None:
            self.car_position = np.array([self.car_x, self.car_y])
            x = np.cos(self.car_yaw)
            y = np.sin(self.car_yaw)
            self.car_direction = np.array([x, y])

        mask_is_left = np.ones(len(self.cones_left_raw), dtype=bool)
        mask_is_right = np.ones(len(self.cones_right_raw), dtype=bool)

        cones_left_adjusted = self.cones_left_raw - self.car_position
        cones_right_adjusted = self.cones_right_raw - self.car_position

        mask_is_left[np.argsort(np.linalg.norm(cones_left_adjusted, axis=1))[5:]] = False
        mask_is_right[np.argsort(np.linalg.norm(cones_right_adjusted, axis=1))[5:]] = False

        cones_left = self.cones_left_raw[mask_is_left]
        cones_right = self.cones_right_raw[mask_is_right]
        cones_unknown = np.row_stack(
            [self.cones_left_raw[~mask_is_left], self.cones_right_raw[~mask_is_right]]
        )

        blue_color = "#7CB9E8"
        yellow_color = "gold"

        for i, c in enumerate(ConeTypes):
            print(c, f"= {i}")

        cones_by_type = [np.zeros((0, 2)) for _ in range(5)]
        cones_by_type[ConeTypes.LEFT] = cones_left
        cones_by_type[ConeTypes.RIGHT] = cones_right
        cones_by_type[ConeTypes.UNKNOWN] = cones_unknown

        out = planner.calculate_path_in_global_frame(
            cones_by_type, self.car_position, self.car_direction, return_intermediate_results=True
        )

        (
            path,
            sorted_left,
            sorted_right,
            left_cones_with_virtual,
            right_cones_with_virtual,
            left_to_right_match,
            right_to_left_match,
        ) = out


    def time_check(self):
        def wrapper():
            self.convert_path(None)
        func = wrapper
        t = timeit.Timer(func)
        execution_time = t.timeit(number=100)
        print(f"Execution time: {execution_time/100} seconds")

    def convert_path(self, event):
        converted_path = Path()
        if self.curr_path is not None and self.markers is not None:
            converted_path.header.frame_id = "odom"
            converted_path.header.stamp = self.stamp
            for x,y in zip(self.curr_path.x, self.curr_path.y):
                poseStamp = PoseStamped()
                poseStamp.header.frame_id = "odom"
                poseStamp.header.stamp = self.stamp
                poseStamp.pose.position.x = x
                poseStamp.pose.position.y = y
                converted_path.poses.append(poseStamp)
            self.original_path_pub.publish(converted_path)
            # self.generate_new_path()



    def curr_path_callback(self, curr_path):
        self.curr_path = curr_path
        # rospy.loginfo(self.curr_path)

    def marker_callback(self, markers):
        self.markers = markers
        for marker in self.markers.markers:
            position = np.array([marker.pose.position.x, marker.pose.position.y])
            if marker.color.b == 1.0:
                if self.cones_right_raw.size == 0:  # Initialize if empty
                    self.cones_right_raw = position
                else:
                    self.cones_right_raw = np.vstack((self.cones_right_raw, position))
            else:
                if self.cones_left_raw.size == 0:  # Initialize if empty
                    self.cones_left_raw = position
                else:
                    self.cones_left_raw = np.vstack((self.cones_left_raw, position))

    def odom_callback(self, odom):
        self.stamp = odom.header.stamp
        self.car_x = odom.pose.pose.position.x
        self.car_y = odom.pose.pose.position.y
        self.car_yaw = odom.pose.pose.orientation.z
        self.car_quart = quaternion_from_euler(0, 0, self.car_yaw)

        self.br.sendTransform((self.car_x, self.car_y, 0),self.car_quart,self.stamp,"test","odom")


if __name__ == "_main_":    
    try:
        node = PathPlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass