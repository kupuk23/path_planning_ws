import numpy as np
from fsd_path_planning.utils.math_utils import unit_2d_vector_from_angle, rotate
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning import PathPlanner, MissionTypes, ConeTypes
import random
import matplotlib.pyplot as plt

import random


#TODO 

GENERATE_OUTLIERS = False
UNDETECTED_CONES = False
FALSE_POS_CONES = False
NUM_FALSE_POS = 2
NUM_UNDETECTED = 2
FIRST_TIME = True
update_step = 5  # update the car position to the next 5 points of the path

planner = PathPlanner(MissionTypes.trackdrive)


def remove_random_cones(points_inner, points_outer, n):
    # Copy the arrays to avoid modifying the original arrays
    inner_cones = np.copy(points_inner)
    outer_cones = np.copy(points_outer)

    # Randomly select n cones to remove from each array
    inner_indices = random.sample(range(len(inner_cones)), n)
    outer_indices = random.sample(range(len(outer_cones)), n)

    # Remove the selected cones from each array
    inner_cones = np.delete(inner_cones, inner_indices, axis=0)
    outer_cones = np.delete(outer_cones, outer_indices, axis=0)

    return inner_cones, outer_cones


def swap_cones(cones_left, cones_right, n):
    # Get the indices of the cones to be swapped
    swap_indices = np.random.choice(len(cones_left), n, replace=False)

    # Swap the cones between left and right arrays
    for index in swap_indices:
        temp = cones_left[index].copy()
        cones_left[index] = cones_right[index]
        cones_right[index] = temp

    return cones_left, cones_right


def get_new_cones(map, car_pos, car_dir, path):
    """
    This function will be called when the car has updated its position and direction,
    The new cones will be generated around the car radius.
    """
    pass


def update_car_pose(path):
    """
    This function will update the car position and direction to the next 5 points of the path
    """
    path_x = path[:, 1]
    path_y = path[:, 2]
    new_position = np.array([path_x[update_step], path_y[update_step]])
    dx = np.diff(
        path_x[update_step : update_step + 3]
    )  # get the gradient with the next 3 points
    dy = np.diff(path_y[update_step : update_step + 3])
    new_direction = np.array([dx[0], dy[0]])
    return new_position, new_direction


car_position = np.array([0.0, 0.0])
car_direction = np.array([0.0, -1.0])
phi_inner = np.arange(0, np.pi / 1.8, np.pi / 18)
phi_outer = np.arange(0, np.pi / 2, np.pi / 20)
phi_test = np.arange(0, -np.pi / 2, -np.pi / 20)

rng = np.random.default_rng(5)


outlier = np.arange(0, np.pi / 10, np.pi / 50)
outliers = unit_2d_vector_from_angle(outlier) * 13
points_inner = unit_2d_vector_from_angle(phi_inner) * 9  # BLUE OR LEFT CONES
points_test = unit_2d_vector_from_angle(phi_test) * 5
# points_inner = np.concatenate((points_inner, outliers), axis=0)
# points_inner = np.concatenate((points_inner, outliers_random), axis=0)
outliers = unit_2d_vector_from_angle(outlier) * 8
points_outer = unit_2d_vector_from_angle(phi_outer) * 12
# points_outer = np.concatenate((points_outer, outliers), axis=0)

# outliers_cones_inner = np.array([[5,4], [6,5], [4,2], [4,1]])
# outliers_cones_outer = np.array([[5,-1.3], [6,0], [7,0.5],[8,0.7],[9,1]])


center = np.mean((points_inner[:2] + points_outer[:2]) / 2, axis=0)
points_inner -= center
points_outer -= center

cones_left_raw = points_inner
cones_right_raw = points_outer


map = np.vstack((points_inner, points_outer))

for step in range(7):

    if not FIRST_TIME:
        car_position, car_direction = update_car_pose(path)
        # get_new_cones()

    if UNDETECTED_CONES and FIRST_TIME:
        cones_left_raw, cones_right_raw = remove_random_cones(
            cones_left_raw, cones_right_raw, NUM_UNDETECTED
        )

    if FALSE_POS_CONES:
        cones_left_raw, cones_right_raw = swap_cones(
            cones_left_raw, cones_right_raw, NUM_FALSE_POS
        )

    if GENERATE_OUTLIERS and FIRST_TIME:
        cones_left_flipped = cones_right_raw[:].copy()
        cones_left_flipped[:, 1] += 10
        cones_left_flipped[:, 0] += 10

        cones_right_flipped = cones_left_raw[:].copy()
        cones_right_flipped[:, 1] += 7
        cones_right_flipped[:, 0] += 7
        cones_left_raw = np.concatenate((cones_left_raw, cones_right_flipped))
        cones_right_raw = np.concatenate((cones_right_raw, cones_left_flipped))

    rng = np.random.default_rng()
    rng.shuffle(cones_left_raw)
    rng.shuffle(cones_right_raw)

    mask_is_left = np.ones(len(cones_left_raw), dtype=bool)
    mask_is_right = np.ones(len(cones_right_raw), dtype=bool)

    cones_left = (
        []
    )  # for demonstration purposes, we will only keep the color of the first 4 cones
    cones_right = []

    cones_left_adjusted = cones_left_raw - car_position
    cones_right_adjusted = cones_right_raw - car_position

    mask_is_left[np.argsort(np.linalg.norm(cones_left_adjusted, axis=1))[5:]] = False
    mask_is_right[np.argsort(np.linalg.norm(cones_right_adjusted, axis=1))[5:]] = False

    cones_left = cones_left_raw[mask_is_left]
    cones_right = cones_right_raw[mask_is_right]
    cones_unknown = np.row_stack(
        [cones_left_raw[~mask_is_left], cones_right_raw[~mask_is_right]]
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
        cones_by_type, car_position, car_direction, return_intermediate_results=True
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

    plt.clf()
    plt.scatter(cones_left[:, 0], cones_left[:, 1], c=blue_color, label="left")
    plt.scatter(cones_right[:, 0], cones_right[:, 1], c=yellow_color, label="right")
    plt.scatter(cones_unknown[:, 0], cones_unknown[:, 1], c="k", label="unknown")

    plt.legend()

    plt.plot(
        [car_position[0], car_position[0] + car_direction[0]],
        [car_position[1], car_position[1] + car_direction[1]],
        c="k",
    )
    plt.title("Computed path")
    plt.plot(*path[:, 1:3].T)  # take path[1] and path[2] to get the x y of the paths

    plt.axis("equal")

    plt.show()

    FIRST_TIME = False
