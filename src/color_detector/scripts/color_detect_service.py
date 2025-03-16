#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String
from color_detector.srv import ColorDetector, ColorDetectorResponse
import numpy as np

# Configuration parameters - pre-computed reusable values
CONFIG = {
    'min_area': 3000,  # Minimum area threshold
    'rect_ratio': 0.85,  # Rectangle ratio threshold
    'resize_width': 640,  # Processing image width
    'blur_kernel_size': 5,  # Median filter kernel size
    'morph_kernel': np.ones((5, 5), np.uint8),  # Pre-computed kernel for morphological operations
    'image_total_area': 640*480,
    'fps_calc_interval': 30,  # Calculate FPS after processing this many frames
    'colors': {
        'red': {
            'ranges': [
                {'lower': np.array([0, 120, 50], dtype=np.uint8), 'upper': np.array([10, 255, 255], dtype=np.uint8)},
                {'lower': np.array([170, 120, 50], dtype=np.uint8), 'upper': np.array([180, 255, 255], dtype=np.uint8)}
            ],
            'bgr': (0, 0, 255)
        },
        'green': {
            'ranges': [
                {'lower': np.array([35, 120, 50], dtype=np.uint8), 'upper': np.array([85, 255, 255], dtype=np.uint8)}
            ],
            'bgr': (0, 255, 0)
        }
    },
    # Rectangle size filtering conditions
    'rect_filter': {
        'width_range': (50, 120),   # Width range (pixels)
        'height_range': (50, 160)   # Height range (pixels)
    }
}

# Pre-created common font
FONT = cv2.FONT_HERSHEY_SIMPLEX

class ImageProcessingServer:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('detect_color')
        
        # Create CV bridge
        self.bridge = CvBridge()
        
        # Create service
        self.service = rospy.Service('/detect_color', ColorDetector, self.handle_request)
        
        # Subscriber and received image
        self.image_subscriber = None
        self.current_image = None
        self.received_image = False
        
        # Create an empty default response
        self.empty_response = ColorDetectorResponse()
        
        rospy.loginfo("Image processing service started, waiting for requests...")
    
    def image_callback(self, msg):
        """
        Callback function when an image is received
        """
        try:
            # Convert ROS image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image
            self.received_image = True
            rospy.loginfo("Image received")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", e)

    def create_color_mask(self, hsv_img, color_ranges):
        """Create color mask, optimized version"""
        # Handle common case of single color range
        if len(color_ranges) == 1:
            return cv2.inRange(hsv_img, color_ranges[0]['lower'], color_ranges[0]['upper'])
        
        # Handle multiple color ranges
        mask = cv2.inRange(hsv_img, color_ranges[0]['lower'], color_ranges[0]['upper'])
        for range_dict in color_ranges[1:]:
            mask |= cv2.inRange(hsv_img, range_dict['lower'], range_dict['upper'])
        return mask

    def process_image(self, image):
        """
        Process the image to detect colored objects
        """
        # Apply median blur for noise reduction
        filter_image = cv2.medianBlur(image, CONFIG['blur_kernel_size'])

        # Get image dimensions
        img_height, img_width = image.shape[:2]
        img_center_x = img_width / 2
        img_center_y = img_height / 2

        # Convert to HSV color space
        hsv = cv2.cvtColor(filter_image, cv2.COLOR_BGR2HSV)

        result_json = {}

        # Only iterate through possible colors
        for color_name, color_config in CONFIG['colors'].items():
            # Create color mask
            mask = self.create_color_mask(hsv, color_config['ranges'])
            
            # Morphological operations to improve mask quality - using pre-computed kernel
            kernel = CONFIG['morph_kernel']
            # Combine operations to reduce temporary array creation
            mask = cv2.morphologyEx(
                cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel),
                cv2.MORPH_CLOSE, kernel
            )
            
            # Find contours - use CHAIN_APPROX_SIMPLE to reduce memory usage
            # Compatible with different versions of OpenCV
            if cv2.__version__.startswith('3.'):
                # OpenCV 3.x
                _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            else:
                # OpenCV 4.x and above
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process contours
            contour_result = self.process_contours(contours, img_center_x, img_center_y)
            
            # Only build result when a contour is detected
            if contour_result:
                cx, cy, area = contour_result

                result_json = {
                    "color": color_name,
                    "position": {"x": cx, "y": cy},
                    "area": float(area),
                }
                # Exit loop after finding a matching color block
                break

        return result_json
    
    def process_contours(self, contours, img_center_x, img_center_y):
        """Process detected contours to find matching rectangles"""
        if not contours:  # Quick check for contours
            return None

        # Get rectangle size filtering range
        width_range = CONFIG['rect_filter']['width_range']
        height_range = CONFIG['rect_filter']['height_range']

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > CONFIG['min_area']:
                # Only approximate large contours exceeding the threshold
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.08 * perimeter, True)
                rospy.loginfo("okkkkkk")
                # Only process rectangles (shapes with 4 vertices)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    rospy.loginfo("len(approx) == 4")

                    # Check if it meets rectangle block size requirements
                    is_width_valid = width_range[0] <= w <= width_range[1]
                    is_height_valid = height_range[0] <= h <= height_range[1]
                    is_height_greater_than_width = h >= w

                    if is_width_valid and is_height_valid and is_height_greater_than_width:
                        rospy.loginfo("is_width_valid")

                        # Calculate center point
                        center_x = x + w // 2
                        center_y = y + h // 2
                    
                        # Calculate coordinates relative to image center (using common computer vision coordinate system)
                        cx = center_x - img_center_x
                        cy = img_center_y - center_y  # Note y-axis direction flip

                        return (cx, cy, area)

        return None

    def handle_request(self, req):
        """
        Handle service requests
        """
        # Make sure to clean up any existing subscriber before starting
        if self.image_subscriber:
            rospy.loginfo("Cleaning up previous subscriber")
            self.image_subscriber.unregister()
            self.image_subscriber = None
            # Give ROS time to fully clean up subscriber
            rospy.sleep(0.1)
        
        try:
            topic_name = req.camera_topic
            rospy.loginfo("Received image processing request, topic name: %s", topic_name)
            
            # Reset image reception status
            self.received_image = False
            self.current_image = None
            
            # Create subscriber and wait for it to be properly registered
            self.image_subscriber = rospy.Subscriber(
                topic_name, Image, self.image_callback, queue_size=1)
            # Give ROS time to properly set up the subscriber
            rospy.sleep(0.2)
            
            # Wait for image reception, set timeout
            timeout = rospy.Duration(5.0)  # 5 second timeout
            start_time = rospy.Time.now()
            
            while not self.received_image and (rospy.Time.now() - start_time) < timeout:
                rospy.sleep(0.1)
            
            # Make a local copy of the image
            local_image = None
            if self.received_image and self.current_image is not None:
                local_image = self.current_image.copy()
            
            # Cancel subscription - explicitly do this before processing
            if self.image_subscriber:
                rospy.loginfo("Unregistering subscriber")
                self.image_subscriber.unregister()
                self.image_subscriber = None
                # Give ROS time to clean up
                rospy.sleep(0.1)
            
            # Check if image was received
            if local_image is None:
                rospy.logwarn("Could not receive image within timeout period")
                return self.empty_response
            
            # Process image (using local copy)
            result_json = self.process_image(local_image)

            if result_json:
                rospy.loginfo("Image processing complete")
                rospy.loginfo(result_json)
                # Return detected color block information
                return ColorDetectorResponse(
                    color=result_json['color'],
                    x=result_json['position']['x'], 
                    y=result_json['position']['y'], 
                    area=result_json['area'], 
                )
            else:
                rospy.loginfo("No matching color blocks detected")
                return self.empty_response
                
        except Exception as e:
            rospy.logerr("Image processing error: %s", str(e))
            # Make sure to clean up subscriber in case of error
            if self.image_subscriber:
                try:
                    self.image_subscriber.unregister()
                    self.image_subscriber = None
                except:
                    pass
            return self.empty_response

if __name__ == '__main__':
    server = ImageProcessingServer()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Server shutdown")