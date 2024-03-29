cmake_minimum_required(VERSION 2.8.3)
project(deep_detection)


#===============================================================================
# C A T K I N   C O N F I G U R A T I O N

find_package(catkin REQUIRED COMPONENTS
  rospy
  message_generation
  geometry_msgs
  dynamic_reconfigure
)

#===============================================================================
# G E N E R A T E   M E S S A G E S   &   S E R V I C E S

catkin_python_setup()

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    message_generation
    image_transport
    cv_bridge
    roslaunch
    )

catkin_package(
    CATKIN_DEPENDS
    roscpp
    std_msgs
    message_runtime
    image_transport
    cv_bridge
    roslaunch
)

#===============================================================================
# I N C L U D E   F I L E S

include_directories(
  ${catkin_INCLUDE_DIRS}
)

#===============================================================================
# I N S T A L L
catkin_install_python(PROGRAMS bin/deepDetection.py
 DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

#===============================================================================
# U N I T   T E S T S

# Add folders to be run by python nosetests
#catkin_add_nosetests(test)
