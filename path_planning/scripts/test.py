#!/usr/bin/env python3
from fsd_path_planning.utils.math_utils import unit_2d_vector_from_angle, rotate
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning import PathPlanner, MissionTypes, ConeTypes
import rospy
import numpy as np
import timeit
from std_msgs.msg import String, Int16, Float32
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import Odometry, Path
from fs_msgs.msg import PlannedPath
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Pose
import tf
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import threading
# Rest of the code

planner = PathPlanner(MissionTypes.trackdrive)


class PathPlanner:
    def __init__(self):
        rospy.init_node("path_planner", anonymous=True)
        rospy.Subscriber("/slam/output/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/slam/output/markers_map", MarkerArray, self.marker_callback)
        rospy.Subscriber( "/navigation/speed_profiler/path", PlannedPath, self.curr_path_callback)
        # self.pub = rospy.Publisher('/kthfs/result', Float32, queue_size=10)

        def wrapper():
            self.convert_path(None)
            
        self.original_path_pub = rospy.Publisher("/original_path", Path, queue_size=10)
        self.generated_path_pub = rospy.Publisher("/generated_path", Path, queue_size=10)
        self.generate_heading_pub = rospy.Publisher("/heading", Marker, queue_size=10)
        self.yaw_test = rospy.Publisher("/yaw_test", Float32, queue_size=10)
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

    def generate_new_path(self, car_pos, car_dir, cones_left_raw, cones_right_raw):
        if not cones_left_raw.size:
            return
        
        mask_is_left = np.ones(len(cones_left_raw), dtype=bool)
        mask_is_right = np.ones(len(cones_right_raw), dtype=bool)

        cones_left_adjusted = cones_left_raw - car_pos
        cones_right_adjusted = cones_right_raw - car_pos

        mask_is_left[np.argsort(np.linalg.norm(cones_left_adjusted, axis=1))[5:]] = False
        mask_is_right[np.argsort(np.linalg.norm(cones_right_adjusted, axis=1))[5:]] = False

        cones_left = cones_left_raw[mask_is_left]
        cones_right = cones_right_raw[mask_is_right]

        cones_unknown = np.row_stack(
            [cones_left[~mask_is_left], cones_right[~mask_is_right]]
        )

        blue_color = "#7CB9E8"
        yellow_color = "gold"

        # for i, c in enumerate(ConeTypes):
        #     print(c, f"= {i}")

        cones_by_type = [np.zeros((0, 2)) for _ in range(5)]
        cones_by_type[ConeTypes.LEFT] = cones_left
        cones_by_type[ConeTypes.RIGHT] = cones_right
        cones_by_type[ConeTypes.UNKNOWN] = cones_unknown

        out = planner.calculate_path_in_global_frame(
            cones_by_type, car_pos, car_dir, return_intermediate_results=True
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

        generated_path = Path()
        generated_path.header.frame_id = "odom"
        generated_path.header.stamp = self.stamp
        for x,y in zip(path[1], path[2]):
            poseStamp = PoseStamped()
            poseStamp.header.frame_id = "odom"
            poseStamp.header.stamp = self.stamp
            poseStamp.pose.position.x = x
            poseStamp.pose.position.y = y
            generated_path.poses.append(poseStamp)
        self.generated_path_pub.publish(generated_path)


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



    def curr_path_callback(self, curr_path):
        self.curr_path = curr_path
        # rospy.loginfo(self.curr_path)

    def marker_callback(self, markers):
        self.markers = markers
        marker_array = MarkerArray()

        self.cones_right_raw = np.array([])
        self.cones_left_raw = np.array([])
        for i,marker in enumerate(self.markers.markers):
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
        quart_angle = [odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w]
        self.car_yaw = euler_from_quaternion(quart_angle)[2]
        self.generate_marker(odom)
        self.car_position = np.array([self.car_x, self.car_y])
        self.car_direction = unit_2d_vector_from_angle(self.car_yaw)
        self.generate_new_path(self.car_position, self.car_direction, self.cones_left_raw, self.cones_right_raw)
        
    def generate_marker(self, odom):
        heading = Marker()
        heading.header.frame_id = "odom"
        heading.header.stamp = odom.header.stamp
        heading.type = 0
        heading.id = 100
        heading.scale.x = 0.7
        heading.scale.y = 0.15
        heading.scale.z = 0.15
        heading.pose.position.x = self.car_x
        heading.pose.position.y = self.car_y
        heading.pose.position.z = 0
        heading.pose.orientation.x = odom.pose.pose.orientation.x
        heading.pose.orientation.y = odom.pose.pose.orientation.y
        heading.pose.orientation.z = odom.pose.pose.orientation.z
        heading.pose.orientation.w = odom.pose.pose.orientation.w
        heading.color.a = 1.0
        heading.color.r = 1.0

        self.generate_heading_pub.publish(heading)

if __name__ == "__main__":    
    try:
        node = PathPlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass