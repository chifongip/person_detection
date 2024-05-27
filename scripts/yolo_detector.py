#!/usr/bin/env python

import sys 
import cv2
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs2
if (not hasattr(rs2, 'intrinsics')):
    import pyrealsense2.pyrealsense2 as rs2

import rospy
from sensor_msgs.msg import CameraInfo, Image
from yolo_v8_ros_msgs.msg import BoundingBoxes, BoundingBox

_noise_limit = 100
_diff = 500
_bin = [i * _diff for i in range(0, 9)]

class yoloDetector:
    def __init__(self) -> None:
        # Load parameters 
        weight_path = rospy.get_param('~weight_path', '')
        info_topic = rospy.get_param('~info_topic', '/camera/depth/camera_info')
        depth_topic = rospy.get_param('~depth_topic', '/camera/depth/image_rect_raw')
        image_topic = rospy.get_param('~image_topic', '/camera/color/image_raw')
        self.conf = rospy.get_param('~conf', '0.6')
        self.imgsz = rospy.get_param('~imgsz', '640')
        if (rospy.get_param('~use_cpu', 'true')):
            self.device = 'cpu'
        else:
            self.device = 'cuda'

        self.model = YOLO(weight_path, task="detect")
        if '.pt' in weight_path:
            self.model.fuse()

        self.getDepthCameraInfoStatus = False
        self.getDepthImageStatus = False
        self.getImageStatus = False

        self.intrinsics = None
        self.depth_image = None

        # Subscriber 
        self.info_sub = rospy.Subscriber(info_topic, CameraInfo, self.cameraInfoCallback, queue_size=1)
        self.depth_sub = rospy.Subscriber(depth_topic, Image, self.depthCallback, queue_size=1, buff_size=2**24)
        self.image_sub = rospy.Subscriber(image_topic, Image, self.imageCallback, queue_size=1, buff_size=2**24)

        # Publisher 
        self.image_pub = rospy.Publisher('/yolov8/detection_image', Image, queue_size=1)
        self.bbox_pub = rospy.Publisher('/yolov8/bounding_boxes', BoundingBoxes, queue_size=1)

        while (not self.getDepthCameraInfoStatus) or (not self.getDepthImageStatus) or (not self.getImageStatus):
            rospy.loginfo("Waiting for image.")
            rospy.sleep(1)


    def cameraInfoCallback(self, cameraInfo) -> None:
        "Subscribe to depth camera info."
        self.getDepthCameraInfoStatus = True
        if self.intrinsics:
            return
        self.intrinsics = rs2.intrinsics()
        self.intrinsics.width = cameraInfo.width
        self.intrinsics.height = cameraInfo.height
        self.intrinsics.ppx = cameraInfo.K[2]
        self.intrinsics.ppy = cameraInfo.K[5]
        self.intrinsics.fx = cameraInfo.K[0]
        self.intrinsics.fy = cameraInfo.K[4]
        if cameraInfo.distortion_model == 'plumb_bob':
            self.intrinsics.model = rs2.distortion.brown_conrady
        elif cameraInfo.distortion_model == 'equidistant':
            self.intrinsics.model = rs2.distortion.kannala_brandt4
        self.intrinsics.coeffs = [i for i in cameraInfo.D]


    def depthCallback(self, msg) -> None:
        "Subscribe to depth image."
        self.getDepthImageStatus = True
        self.depth_image = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width, -1)
        

    def imageCallback(self, msg) -> None:
        "Subscribe to image and perform YOLO detection."
        self.getImageStatus = True
        color_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # results = self.model.predict(color_image, 
        #                             show=False, 
        #                             verbose=False, 
        #                             conf=self.conf, 
        #                             imgsz=self.imgsz, 
        #                             device=self.device)

        results = self.model.track(color_image, 
                                    show=False, 
                                    verbose=False, 
                                    conf=self.conf, 
                                    imgsz=self.imgsz, 
                                    device=self.device,
                                    tracker="botsort.yaml")

        self.detectResult(results, msg.header.stamp, msg.header.frame_id, msg.height, msg.width)


    def detectResult(self, results, image_stamp, frame_id, height, width) -> None:
        "Publish detection result."
        frame = results[0].plot()
        fps = 1000.0/ results[0].speed['inference']
        cv2.putText(frame, f'FPS: {int(fps)}', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        bounding_boxes = BoundingBoxes()
        bounding_boxes.header.stamp = rospy.Time.now()
        bounding_boxes.header.frame_id = frame_id
        bounding_boxes.image_header.stamp = image_stamp
        bounding_boxes.image_header.frame_id = frame_id

        # line = ''

        for result in results[0].boxes:
            if results[0].names[result.cls.item()] == "person":
                bounding_box = BoundingBox()
                bounding_box.object_class = results[0].names[result.cls.item()]
                bounding_box.confidence = result.conf.item()
                try:
                    bounding_box.id = np.int64(result.id[0].item())
                except:
                    pass
                bounding_box.xmin = np.int64(result.xyxy[0][0].item())
                bounding_box.ymin = np.int64(result.xyxy[0][1].item())
                bounding_box.xmax = np.int64(result.xyxy[0][2].item())
                bounding_box.ymax = np.int64(result.xyxy[0][3].item())

                pix = ((bounding_box.xmin + bounding_box.xmax) // 2, (bounding_box.ymin + bounding_box.ymax) // 2)
                depth = self.depth_image[pix[1], pix[0]]
                xyz_optical = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [pix[0], pix[1]], depth)
                
                roi = self.depth_image[bounding_box.ymin:bounding_box.ymax, bounding_box.xmin:bounding_box.xmax]
                hist, bins = np.histogram(roi.flatten(), bins=_bin, density=True)
                max_bin = hist.argmax() + 1
                filtered_depth, _size = self.filter_background(roi, max_bin * _diff)
                if _size:
                    avg_depth = filtered_depth.sum() / _size 
                else:
                    avg_depth = 0

                bounding_box.x = xyz_optical[0] / 1000
                bounding_box.y = xyz_optical[1] / 1000
                # bounding_box.z = xyz_optical[2] / 1000
                bounding_box.z = avg_depth / 1000
                bounding_boxes.bounding_boxes.append(bounding_box)

                # line += '\rX: %f, Y: %f , Z: %f\r'% (bounding_box.x, bounding_box.y, bounding_box.z)

                cv2.circle(frame, (pix[0], pix[1]), radius=0, color=(0, 0, 255), thickness=5)

        self.bbox_pub.publish(bounding_boxes)

        # sys.stdout.write(line)
        # sys.stdout.flush()

        image_temp = Image()
        image_temp.header.stamp = image_stamp
        image_temp.header.frame_id = frame_id
        image_temp.height = height
        image_temp.width = width
        image_temp.encoding = 'bgr8'
        image_temp.data = np.array(frame).tobytes()
        image_temp.step = width * 3
        self.image_pub.publish(image_temp)


    def filter_background(self, roi, max_depth=4000):
        # Anything further than 4000mm, we consider it as background
        # Anything less than 100mm is consider noise
        ret_val = np.ma.masked_greater(roi, max_depth)
        ret_val = np.ma.masked_less(ret_val, _noise_limit)
        unique, counts = np.unique(ret_val.mask, return_counts=True)
        _dict = dict(zip(unique, counts))
        if False in _dict:
            return ret_val, _dict[False]
        else:
            return ret_val, 0

        
if __name__ == '__main__':
    try:
        rospy.init_node('yolo_detector', anonymous=True)
        yolo_detector = yoloDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass