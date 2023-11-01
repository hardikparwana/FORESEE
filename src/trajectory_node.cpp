#include <iostream>
#include <math.h>
#include "rclcpp/rclcpp.hpp"
#include "dasc_msgs/msg/di_trajectory.hpp"

class TrajectoryPublisher: public rclcpp::Node {
    public:
        TrajectoryPublisher(): Node("trajectory_node"){

            robot_trajectory_pub = this->create_publisher<dasc_msgs::msg::DITrajectory>("trajectory",10);
            timer_ = this->create_wall_timer( std::chrono::duration<double>(1.0/) )

            generate_trajectory();


        }

        void generate_trajectory(){
            float t = 0.0;
            float tf = 3.0;
            float R = 1.5;


            // circle trajectory
            float omega = 2.0;
            float v = 0.6;

            dasc_msgs::msg::DITrajectory msg;
            msg.dt = delta_t;

            for (int i=0; i<int(tf/delta_t); i++){

                float theta = omega * t;

                geometry_msgs::Pose pose;
                geometry_msgs::Twist twist;
                geometry_msgs::Accel accel;

                float theta = omega*t;
                pose.position.x = R * cos(theta);
                pose.position.y = R * sin(theta);
                pose.position.z = 0.0;
                pose.orientation.x = 0.0;
                pose.orientation.y = 0.0;
                pose.orientation.w = sin(theta/2);
                pose.orientation.z = cos(theta/2);
                msg.poses.push_back(pose);

                twist.linear.x = 0.0;
                twist.linear.y = 0.0;
                twist.linear.z = 0.0;
                twist.angular.x = 0.0;
                twist.angular.y = 0.0;
                twist.angular.z = 0.0;
                msg.twists.push_back(twist);

                msg.accelerations.push_back(accel);

            }

            robot_trajectpry_pub_->publish(msg);


            
        }

    private:

        rclcpp::Publisher<dasc_msgs::msg::DITrajectory>::SharedPtr robot_trajectory_pub_;
        float delta_t = 0.05;
        float update_frequency;
        rclcpp::TimerBase::SharedPtr timer_;


}

int main(int argc, char **argv){
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TrajectoryPublisher>);
    rclcpp::shutdown();
    return 0;
}