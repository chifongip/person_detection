# person_detection  
A ROS package for person detection based on YOLOv8 and Intel RealSense depth camera.  

## Installation  
### Clone the related packages
```
git clone https://github.com/chifongip/person_detection.git
git clone https://github.com/chifongip/yolo_v8_ros_msgs.git
```

### Install [YOLOv8](https://github.com/ultralytics/ultralytics.git)
pip install the ultralytics package including all requirements in a Python>=3.8 environment with PyTorch>=1.8.
```
pip install ultralytics
```

### Install [RealSense ROS wrapper](https://github.com/IntelRealSense/realsense-ros.git)

#### Install librealsense2 debian package:
- Register the server's public key:
```
sudo mkdir -p /etc/apt/keyrings 
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
```

- Make sure apt HTTPS support is installed:
`sudo apt-get install apt-transport-https`

- Add the server to the list of repositories:
```
echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
sudo tee /etc/apt/sources.list.d/librealsense.list
sudo apt-get update
```

- Install the libraries (see section below if upgrading packages):  
  `sudo apt-get install librealsense2-dkms`  
  `sudo apt-get install librealsense2-utils`  
  The above two lines will deploy librealsense2 udev rules, build and activate kernel modules, runtime library and executable demos and tools.  

- Install the developer and debug packages:  
  `sudo apt-get install librealsense2-dev`  
  `sudo apt-get install librealsense2-dbg`  

#### Install Intel RealSense ROS from Sources
- Clone the latest Intel RealSense ROS from here into 'catkin_ws/src/':
```
git clone https://github.com/IntelRealSense/realsense-ros.git
```
- Make sure that the ros package ddynamic_reconfigure is installed:
```
sudo apt-get install ros-noetic-ddynamic-reconfigure
```
- Build your workspace:
```
catkin_make -DCATKIN_ENABLE_TESTING=False -DCMAKE_BUILD_TYPE=Release
```

#### Install python wrapper  
```
pip install pyrealsense2
```  

## Parameters 
- **weight_path:** path to the weight file.
- **info_topic:** camera_info topic of the depth camera.
- **depth_topic:** depth image topic.
- **image_topic:**  color image topic.
- **conf:** sets the minimum confidence threshold for detections.
- **imgsz:** defines the image size for inference.
- **use_cpu:** specifies the device for inference.

## Usage
```
roslaunch person_detection yolo_detector.launch
```