#
# Odom with and array of cone arrays (cone batches) that are detected between current odom and the next
#


# Extra information on the message
# header.stamp: time of the message
# header.frame_id: frame in which the cones are given
std_msgs/Header header
string child_frame_id

# Odometry of the vehicle before the first cone in the array
nav_msgs/Odometry odom

# Array of all cone_batches detected after current odom, but before next odom, oldest first
# (vehicle frame)
Cones[] cone_batches

# Relative time since last odom, maps index to index to respective cone_batch
float64[] dt

# Example:
# |(odom) ------------- |(next odom) ------ ...      // next odom is in next message
#   |(cone1)  |(cone2) |(cone3) |(cone4)    ...      // cone1, cone2, cone3 are in this message, and only cone4 is in next message 


# Example:
# |(odom) ------------- |(next odom) ------ ...      // next odom is in next message
#   |(cone1)  |(cone2)   |(cone3) |(cone4)  ...      // cone1, cone2 are in this message, and cone3, cone4 are in next message 

