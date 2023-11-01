cmake_minimum_required(VERSION 3.5)
project(foresee)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(dasc_msgs REQUIRED)

include_directories(include)


# this needed to tell colcon to place launch files in the share folder
install(DIRECTORY launch configs
  DESTINATION share/${PROJECT_NAME})
# install(DIRECTORY scripts DESTINATION share/${PROJECT_NAME})


add_executable(trajectory_node nodes/trajectory_node.cpp)
ament_target_dependencies(trajectory_node rclcpp dasc_msgs)
install( TARGETS
    trajectory_node
    DESTINATION lib/${PROJECT_NAME}
)
