cmake_minimum_required(VERSION 3.8)
project(fs_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(actionlib_msgs REQUIRED)

find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
    "msg/ClassifiedBoundingBox.msg"
    "msg/ClassifiedBoundingBoxes.msg"
    "msg/ClassifiedBoundingBoxesImage.msg"
    "msg/Cone.msg"
    "msg/Cones.msg"
    "msg/ConeStats.msg"
    "msg/ConeWithStats.msg"
    "msg/ConesWithStats.msg"
    "msg/ConeImage.msg"
    "msg/ConeImages.msg"
    "msg/SlamState.msg"
    "msg/SlamInput.msg"
    "msg/ControllerOutput.msg"
    "msg/PIDControlled.msg"
    "msg/PlannedPath.msg"
    "msg/SbgEKFStatus.msg"
    "msg/OutlierConfig.msg"
    "msg/Wheelspeeds.msg"
    "msg/CarCommand.msg"
    "msg/SlamMSE.msg"
    "msg/Track.msg"
    "msg/FSDSCone.msg"
    "msg/FSDSControlCommand.msg"
    "msg/FSDSExtraInfo.msg"
    "msg/FSDSFinishedSignal.msg"
    "msg/FSDSGoSignal.msg"
  DEPENDENCIES std_srvs
  DEPENDENCIES std_msgs
  DEPENDENCIES geometry_msgs
  DEPENDENCIES nav_msgs
  DEPENDENCIES sensor_msgs
  DEPENDENCIES actionlib_msgs
  DEPENDENCIES diagnostic_msgs
 )


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()