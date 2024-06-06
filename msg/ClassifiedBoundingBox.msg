#
# Description of a cone bounding box for image detection
#

# Color of the detected cone
uint8 UNDEFINED = 0
uint8 YELLOW = 1
uint8 BLUE = 2
uint8 SMALL_ORANGE = 3
uint8 BIG_ORANGE = 4

uint8 color

# Coordinates of the bounding box in the image
int64 xmin
int64 ymin
int64 xmax
int64 ymax

# Confidence in the detection
float64 probability
