#!/usr/bin/env python

import rospy

from visualization_msgs.msg import MarkerArray, Marker 
from yolo_v8_ros_msgs.msg import BoundingBox, BoundingBoxes


class markerArray:
    def __init__(self) -> None:
        self.bbox_sub = rospy.Subscriber('/yolov8/bounding_boxes', BoundingBoxes, self.bboxCallback, queue_size=1)
        self.markerarray_pub = rospy.Publisher('/yolov8/marker_array', MarkerArray, queue_size=1)
    

    def bboxCallback(self, msg) -> None:
        "Publish markers."
        marker_array = MarkerArray()
        if msg.bounding_boxes:
            for bbox in msg.bounding_boxes:
                marker = Marker()
                marker.header.stamp = rospy.Time.now()
                marker.header.frame_id = "camera_depth_optical_frame"
                marker.ns = "person"
                marker.id = bbox.id
                marker.type = 2
                marker.action = 0
                marker.pose.position.x = bbox.x
                marker.pose.position.y = bbox.y
                marker.pose.position.z = bbox.z
                marker.pose.orientation.x = 0
                marker.pose.orientation.y = 0
                marker.pose.orientation.z = 0
                marker.pose.orientation.w = 1
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                marker.color.r = 1
                marker.color.g = 0
                marker.color.b = 0
                marker.color.a = 1
                marker.lifetime = rospy.Duration(0.1)
                marker.frame_locked = True
                marker_array.markers.append(marker)
                
            self.markerarray_pub.publish(marker_array)


if __name__ == '__main__':
    try:
        rospy.init_node('marker_array', anonymous=True)
        marker_array = markerArray()
        rospy.loginfo("Publish Marker to Rviz.")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass