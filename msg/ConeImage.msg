#
# Image of a cone and its position in space
#

# 2D-position of the cone (expressed in the frame stated in the Cones message)
float64 x
float64 y
float64 z

# Image of the cone
sensor_msgs/Image image

# Covariance on the position [m^2] (2x2 matrix in row-major order)
float64[4] covariance
