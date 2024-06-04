#!/usr/bin/env python3
from fsd_path_planning.utils.math_utils import unit_2d_vector_from_angle, rotate
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning import PathPlanner, MissionTypes, ConeTypes
import rospy
import numpy as np
from numpy import random
import math
import timeit
from std_msgs.msg import String, Int16, Float32
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import Odometry, Path
from fs_msgs.msg import PlannedPath
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Pose
import tf
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import threading
from copy import deepcopy
# Rest of the code

UNCOLORED_CONES = True
planner = PathPlanner(MissionTypes.trackdrive)


class PathPlanner:
    def __init__(self):
        rospy.init_node("path_planner", anonymous=True)
        rospy.Subscriber("/slam/output/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/slam/output/markers_map", MarkerArray, self.marker_callback)
        rospy.Subscriber( "/navigation/speed_profiler/path", PlannedPath, self.curr_path_callback)
        # self.pub = rospy.Publisher('/kthfs/result', Float32, queue_size=10)

            
        self.original_path_pub = rospy.Publisher("/original_path", Path, queue_size=10)
        self.generated_path_pub = rospy.Publisher("/generated_path", Path, queue_size=10)
        self.generate_heading_pub = rospy.Publisher("/heading", Marker, queue_size=10)
        self.uncolored_path_pub = rospy.Publisher("/uncolored_path", Path, queue_size=10)
        self.false_cones_pub = rospy.Publisher("/false_cones", MarkerArray, queue_size=10)
        self.yaw_test = rospy.Publisher("/yaw_test", Float32, queue_size=10)
        rospy.Timer(rospy.Duration(1/2), self.convert_path)
        # rospy.Timer(rospy.Duration(1/2), self.generate_new_path)
        self.br = tf.TransformBroadcaster()

        self.curr_path = None
        
        self.stamp = None
        self.cones_right_raw = np.array([])
        self.cones_left_raw = np.array([])
        self.false_cones = MarkerArray()
        self.false_p_left = MarkerArray()
        print("running")


    def generate_new_path(self, car_pos, car_dir,  cones_left_raw, cones_right_raw, both_cones = True):
        if not cones_left_raw.size:
            return
        
        mask_is_left = np.ones(len(cones_left_raw), dtype=bool)
        mask_is_right = np.ones(len(cones_right_raw), dtype=bool)

        cones_left_adjusted = cones_left_raw - car_pos
        cones_right_adjusted = cones_right_raw - car_pos

        mask_is_left[np.argsort(np.linalg.norm(cones_left_adjusted, axis=1))[5:]] = False
        mask_is_right[np.argsort(np.linalg.norm(cones_right_adjusted, axis=1))[5:]] = False

       


        path = self.run_path_planner(cones_left_raw, cones_right_raw, mask_is_left, mask_is_right, car_pos, car_dir)

        generated_path = Path()
        generated_path.header.frame_id = "odom"
        generated_path.header.stamp = self.stamp
        for pose in path:
            poseStamp = PoseStamped()
            poseStamp.header.frame_id = "odom"
            poseStamp.header.stamp = self.stamp
            poseStamp.pose.position.x = pose[1]
            poseStamp.pose.position.y = pose[2]
            generated_path.poses.append(poseStamp)


        if not both_cones:
            cones_right_raw = np.array([])
            mask_is_left = np.ones(len(cones_left_raw), dtype=bool)
            cones_left_adjusted = cones_left_raw - car_pos
            mask_is_left[np.argsort(np.linalg.norm(cones_left_adjusted, axis=1))[5:]] = False
            mask_is_right = np.zeros(len(cones_right_raw), dtype=bool)

        false_cones = np.array([(marker.pose.position.x, marker.pose.position.y) for marker in self.false_cones.markers])

        path = self.run_path_planner(cones_left_raw, cones_right_raw, mask_is_left, mask_is_right, car_pos, car_dir, false_cones, uncolored=False, both_cones=both_cones)
        
        uncolored_path = Path()
        uncolored_path.header.frame_id = "odom"
        uncolored_path.header.stamp = self.stamp
        for pose in path:
            poseStamp = PoseStamped()
            poseStamp.header.frame_id = "odom"
            poseStamp.header.stamp = self.stamp
            poseStamp.pose.position.x = pose[1]
            poseStamp.pose.position.y = pose[2]
            uncolored_path.poses.append(poseStamp)

        self.uncolored_path_pub.publish(uncolored_path)
        self.generated_path_pub.publish(generated_path)
        

    def run_path_planner(self, cones_left_raw, cones_right_raw, mask_left, mask_right, car_pos, car_dir, false_cones= np.array([]), uncolored = False, both_cones = True):

        if not uncolored:
            cones_left = cones_left_raw[mask_left]
            cones_right = cones_right_raw[mask_right] if both_cones else np.array([])
            
            if false_cones.size == 0:
                cones_unknown = np.row_stack(
                    [cones_left_raw[~mask_left], cones_right_raw[~mask_right]]
                ) if both_cones else cones_left_raw[~mask_left]
            else:
                cones_unknown = np.row_stack(
                    [cones_left_raw[~mask_left], cones_right_raw[~mask_right], false_cones]) if both_cones else np.row_stack([cones_left_raw[~mask_left], false_cones])

        else : 
            cones_left = np.array([])
            cones_right = np.array([])
            if false_cones.size == 0:
                cones_unknown = np.row_stack(
                    [cones_left_raw, cones_right_raw]
                ) if both_cones else cones_left_raw
            else:
                cones_unknown = np.row_stack(
                [cones_left_raw, cones_right_raw, false_cones]
            ) if both_cones else np.row_stack([cones_left_raw, false_cones])
        


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

        return path

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
            self.create_false_cones(self.markers.markers)
            



    def curr_path_callback(self, curr_path):
        self.curr_path = curr_path
        # rospy.loginfo(self.curr_path)

    def marker_callback(self, markers):
        self.markers = markers
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
        
        self.false_cones_pub.publish(self.false_cones)
        

    def create_false_cones(self, cones, generate_radius=4):
        r = cones[0].scale.x
        if random.random() < 0.1:
            # Generate a random radius between 1.2r and 1.5r
            random_radius = random.uniform(1.4 * r, 2.0 * r)

            # Generate a random angle between 0 and 2π
            random_angle = random.uniform(0, 2 * math.pi)
            potential_cones = []
            #iterate through the cones, find the cones inside the car radiusx   
            for cone in cones:
                cone_pos = np.array([cone.pose.position.x, cone.pose.position.y])
                car_pos = np.array([self.car_x, self.car_y])
                if math.dist(cone_pos, car_pos) < generate_radius:
                    potential_cones.append(cone)

                    
            if potential_cones != []:
                new_cone = deepcopy(random.choice(potential_cones))
                # Convert polar coordinates to Cartesian coordinates
                new_x = cone.pose.position.x + random_radius * math.cos(random_angle)
                new_y = cone.pose.position.y + random_radius * math.sin(random_angle)   
                new_cone.pose.position.x = new_x
                new_cone.pose.position.y = new_y
                new_cone.color.a = 1.0
                new_cone.color.r = 1.0
                new_cone.color.g = 1.0
                new_cone.color.b = 1.0
                self.false_cones.markers.append(new_cone)      

    def odom_callback(self, odom):
        
        self.stamp = odom.header.stamp
        self.car_x = odom.pose.pose.position.x
        self.car_y = odom.pose.pose.position.y
        quart_angle = [odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w]
        self.car_yaw = euler_from_quaternion(quart_angle)[2]
        self.generate_marker(odom)
        self.car_position = np.array([self.car_x, self.car_y])
        self.car_direction = unit_2d_vector_from_angle(self.car_yaw)
        self.generate_new_path(self.car_position, self.car_direction, self.cones_left_raw, self.cones_right_raw, both_cones=False)
        
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