#
# Description of a cone
#

# 2D-position of the cone (expressed in the frame stated in the Cones message)
float64 x
float64 y
float64 z

# Color of the cone
uint8 UNDEFINED = 0
uint8 YELLOW = 1
uint8 BLUE = 2
uint8 SMALL_ORANGE = 3
uint8 BIG_ORANGE = 4

uint8 color

# Covariance on the position [m^2] (2x2 matrix in row-major order)
float64[4] covariance

# Confidence in the detection
float64 probability

# ID of the cone 
int32 id
