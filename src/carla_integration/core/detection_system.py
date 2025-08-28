"""
CARLA Detection System
Complete integration system combining SSD detection with CARLA environment
"""

import time
import cv2
import logging
import argparse
import os
import sys
from datetime import datetime

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from .detector import SSDDetector
from .environment import CARLAEnvironment


class CARLADetectionSystem:
    """Complete CARLA detection system"""
    
    def __init__(self, checkpoint_path: str, confidence_threshold: float = 0.5, 
                 carla_host: str = 'localhost', carla_port: int = 2000):
        """Initialize the complete detection system"""
        
        # Initialize detector
        self.detector = SSDDetector(checkpoint_path, confidence_threshold=confidence_threshold)
        
        # Initialize CARLA environment
        self.carla_env = CARLAEnvironment(host=carla_host, port=carla_port)
        
        # System state
        self.running = False
        self.frame_count = 0
        self.start_time = None
        self.total_detections = 0
        
        # Performance tracking
        self.inference_times = []
        self.fps_history = []
        
        logging.info("CARLA Detection System initialized")
    
    def setup(self, vehicle_filter='vehicle.tesla.model3', camera_position='front', 
              weather='ClearNoon', autopilot=True):
        """Setup the detection system"""
        
        logging.info("Setting up CARLA Detection System...")
        
        # List available vehicles
        self.carla_env.list_available_vehicles()
        
        # Spawn vehicle and setup camera
        self.carla_env.spawn_vehicle(vehicle_filter)
        self.carla_env.setup_camera(width=800, height=600, attachment_type=camera_position)
        
        # Set weather and autopilot
        self.carla_env.set_weather(weather)
        self.carla_env.set_vehicle_autopilot(autopilot)
        
        # Wait for first image
        logging.info("Waiting for first camera image...")
        if not self.carla_env.wait_for_first_image(timeout=15.0):
            raise Exception("Failed to receive camera image within timeout")
        
        logging.info("CARLA Detection System ready!")
    
    def run_detection_loop(self, display=True, save_video=False, output_path=None, 
                          max_frames=None, print_detections=True):
        """Main detection loop"""
        
        self.running = True
        self.start_time = time.time()
        
        # Setup video writer
        video_writer = None
        if save_video:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"carla_detection_{timestamp}.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (800, 600))
            logging.info(f"Recording video to: {output_path}")
        
        try:
            logging.info("Starting detection loop... Press 'q' to quit")
            
            while self.running:
                # Get latest image
                image = self.carla_env.get_latest_image()
                if image is None:
                    time.sleep(0.01)
                    continue
                
                # Run detection
                start_inference = time.time()
                boxes, scores, labels = self.detector.detect(image)
                inference_time = time.time() - start_inference
                
                # Update statistics
                self.frame_count += 1
                self.total_detections += len(boxes)
                self.inference_times.append(inference_time)
                
                # Calculate FPS
                elapsed_time = time.time() - self.start_time
                current_fps = self.frame_count / elapsed_time
                self.fps_history.append(current_fps)
                
                # Draw detections
                result_image = self.detector.draw_detections(image.copy(), boxes, scores, labels)
                
                # Add system information overlay
                self._add_info_overlay(result_image, current_fps, inference_time, len(boxes))
                
                # Print detection summary
                if print_detections and len(boxes) > 0:
                    detection_summary = self.detector.get_detection_summary(boxes, scores, labels)
                    logging.info(f"Frame {self.frame_count}: {detection_summary}")
                
                # Display
                if display:
                    cv2.imshow('CARLA SSD Detection', result_image)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):  # Screenshot
                        screenshot_path = f"screenshot_{self.frame_count}.jpg"
                        cv2.imwrite(screenshot_path, result_image)
                        logging.info(f"Screenshot saved: {screenshot_path}")
                
                # Save video frame
                if save_video and video_writer:
                    video_writer.write(result_image)
                
                # Check max frames limit
                if max_frames and self.frame_count >= max_frames:
                    logging.info(f"Reached maximum frames limit: {max_frames}")
                    break
        
        except KeyboardInterrupt:
            logging.info("Detection loop interrupted by user")
        
        finally:
            self.running = False
            if video_writer:
                video_writer.release()
                logging.info(f"Video saved: {output_path}")
            
            if display:
                cv2.destroyAllWindows()
            
            # Print final statistics
            self._print_final_statistics()
    
    def _add_info_overlay(self, image, fps, inference_time, detection_count):
        """Add information overlay to image"""
        
        # Vehicle info
        speed = self.carla_env.get_vehicle_speed()
        location = self.carla_env.get_vehicle_location()
        
        # Performance info
        overlay_text = [
            f"FPS: {fps:.1f}",
            f"Inference: {inference_time*1000:.1f}ms",
            f"Detections: {detection_count}",
            f"Frame: {self.frame_count}",
            f"Speed: {speed:.1f} km/h",
        ]
        
        if location:
            overlay_text.append(f"Location: ({location[0]:.1f}, {location[1]:.1f})")
        
        # Draw overlay
        y_offset = 30
        for i, text in enumerate(overlay_text):
            cv2.putText(image, text, (10, y_offset + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw confidence threshold
        cv2.putText(image, f"Confidence: {self.detector.confidence_threshold:.2f}", 
                   (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def _print_final_statistics(self):
        """Print final performance statistics"""
        
        if self.frame_count == 0:
            return
        
        total_time = time.time() - self.start_time
        avg_fps = self.frame_count / total_time
        avg_inference_time = sum(self.inference_times) / len(self.inference_times)
        
        logging.info("\n" + "="*50)
        logging.info("FINAL STATISTICS")
        logging.info("="*50)
        logging.info(f"Total Runtime: {total_time:.1f}s")
        logging.info(f"Total Frames: {self.frame_count}")
        logging.info(f"Average FPS: {avg_fps:.1f}")
        logging.info(f"Average Inference Time: {avg_inference_time*1000:.1f}ms")
        logging.info(f"Total Detections: {self.total_detections}")
        logging.info(f"Detections per Frame: {self.total_detections/self.frame_count:.1f}")
        logging.info("="*50)
    
    def cleanup(self):
        """Cleanup resources"""
        self.carla_env.cleanup()
        logging.info("Detection system cleanup complete")
