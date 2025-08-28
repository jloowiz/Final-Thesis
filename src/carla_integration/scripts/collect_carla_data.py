#!/usr/bin/env python3
"""
CARLA Dataset Collection Script
Collect annotated images from CARLA for training/evaluation
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import carla
import cv2
import numpy as np


class CARLADataCollector:
    def __init__(self, host='localhost', port=2000):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = None
        self.camera = None
        self.vehicle = None
        self.annotations = []
        
    def setup_world(self, weather_preset='ClearNoon'):
        """Setup CARLA world with specified weather"""
        self.world = self.client.get_world()
        
        # Set weather
        weather = carla.WeatherParameters()
        if weather_preset == 'ClearNoon':
            weather.cloudiness = 0.0
            weather.precipitation = 0.0
            weather.sun_altitude_angle = 70.0
        elif weather_preset == 'Foggy':
            weather.cloudiness = 80.0
            weather.fog_density = 50.0
            weather.precipitation = 0.0
        elif weather_preset == 'Rainy':
            weather.cloudiness = 80.0
            weather.precipitation = 90.0
            weather.precipitation_deposits = 50.0
        
        self.world.set_weather(weather)
        print(f"🌤️  Weather set to: {weather_preset}")
        
    def spawn_vehicle(self, vehicle_filter='vehicle.tesla.model3'):
        """Spawn vehicle in the world"""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(vehicle_filter)[0]
        
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = np.random.choice(spawn_points)
        
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.vehicle.set_autopilot(True)
        
        print(f"🚗 Vehicle spawned: {vehicle_filter}")
        
    def setup_camera(self, camera_position='front'):
        """Setup RGB camera on vehicle"""
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_bp.set_attribute('fov', '90')
        
        # Camera positions
        if camera_position == 'front':
            transform = carla.Transform(carla.Location(x=2.0, z=1.4))
        elif camera_position == 'hood':
            transform = carla.Transform(carla.Location(x=1.0, z=1.0))
        elif camera_position == 'roof':
            transform = carla.Transform(carla.Location(x=0.0, z=2.5))
        
        self.camera = self.world.spawn_actor(camera_bp, transform, attach_to=self.vehicle)
        print(f"📷 Camera attached at: {camera_position}")
        
    def collect_frame_data(self, save_path):
        """Collect single frame with annotations"""
        # Capture image
        image_queue = []
        
        def image_callback(image):
            image_queue.append(image)
        
        self.camera.listen(image_callback)
        
        # Wait for image
        start_time = time.time()
        while len(image_queue) == 0 and time.time() - start_time < 2.0:
            time.sleep(0.01)
        
        if len(image_queue) == 0:
            return None, None
        
        image = image_queue[0]
        self.camera.stop()
        
        # Convert image
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        rgb_image = array[:, :, :3]
        
        # Get world objects for annotation
        annotations = self.get_world_annotations(image.transform)
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        image_filename = f"carla_image_{timestamp}.png"
        image_path = os.path.join(save_path, 'images', image_filename)
        cv2.imwrite(image_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        
        return image_filename, annotations
    
    def get_world_annotations(self, camera_transform):
        """Get bounding box annotations for visible objects"""
        annotations = []
        
        # Get camera calibration
        image_w, image_h = 1280, 720
        fov = 90
        focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))
        
        # Camera matrices
        K = np.array([[focal, 0, image_w/2],
                      [0, focal, image_h/2],
                      [0, 0, 1]])
        
        # Get all actors
        actors = self.world.get_actors()
        
        for actor in actors:
            if actor.type_id.startswith('vehicle.') or actor.type_id.startswith('walker.'):
                # Get bounding box in world coordinates
                bbox = actor.bounding_box
                bbox_vertices = bbox.get_world_vertices(actor.get_transform())
                
                # Project to camera coordinates
                camera_bbox = []
                for vertex in bbox_vertices:
                    # Transform to camera coordinate system
                    vertex_camera = self.world_to_camera(vertex, camera_transform)
                    
                    # Project to image plane
                    if vertex_camera[2] > 0:  # In front of camera
                        x = focal * vertex_camera[0] / vertex_camera[2] + image_w/2
                        y = focal * vertex_camera[1] / vertex_camera[2] + image_h/2
                        camera_bbox.append([x, y])
                
                if len(camera_bbox) >= 4:  # Valid projection
                    # Get 2D bounding box
                    xs = [p[0] for p in camera_bbox]
                    ys = [p[1] for p in camera_bbox]
                    x_min, x_max = max(0, min(xs)), min(image_w, max(xs))
                    y_min, y_max = max(0, min(ys)), min(image_h, max(ys))
                    
                    if x_max > x_min and y_max > y_min:
                        # Determine class
                        if actor.type_id.startswith('vehicle.'):
                            if 'car' in actor.type_id or 'tesla' in actor.type_id:
                                class_name = 'Car'
                            elif 'bus' in actor.type_id:
                                class_name = 'Bus'
                            elif 'truck' in actor.type_id:
                                class_name = 'Truck'
                            elif 'motorcycle' in actor.type_id or 'bike' in actor.type_id:
                                class_name = 'Motorcycle'
                            else:
                                class_name = 'Car'  # Default
                        else:  # walker
                            class_name = 'Person'
                        
                        annotation = {
                            'class_name': class_name,
                            'bbox': [x_min, y_min, x_max - x_min, y_max - y_min],  # COCO format
                            'area': (x_max - x_min) * (y_max - y_min),
                            'actor_id': actor.id
                        }
                        annotations.append(annotation)
        
        return annotations
    
    def world_to_camera(self, world_point, camera_transform):
        """Transform world coordinates to camera coordinates"""
        # Convert to numpy
        world_point = np.array([world_point.x, world_point.y, world_point.z, 1.0])
        
        # Camera transformation matrix
        camera_matrix = np.array(camera_transform.get_matrix())
        camera_to_world = camera_matrix
        world_to_camera = np.linalg.inv(camera_to_world)
        
        # Transform point
        camera_point = world_to_camera.dot(world_point)
        
        # Convert to camera coordinate system (x-right, y-down, z-forward)
        return np.array([camera_point[1], -camera_point[2], camera_point[0]])
    
    def collect_dataset(self, output_path, num_images=1000, scenarios=None):
        """Collect dataset with multiple scenarios"""
        print(f"📊 Collecting {num_images} images to: {output_path}")
        
        # Create directories
        os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'annotations'), exist_ok=True)
        
        collected_images = []
        collected_annotations = []
        
        if scenarios is None:
            scenarios = [
                {'weather': 'ClearNoon', 'vehicle': 'vehicle.tesla.model3'},
                {'weather': 'Foggy', 'vehicle': 'vehicle.audi.etron'},
                {'weather': 'Rainy', 'vehicle': 'vehicle.bmw.grandtourer'}
            ]
        
        images_per_scenario = num_images // len(scenarios)
        
        for scenario_idx, scenario in enumerate(scenarios):
            print(f"\n📋 Scenario {scenario_idx + 1}/{len(scenarios)}: {scenario}")
            
            try:
                # Setup scenario
                self.setup_world(scenario['weather'])
                self.spawn_vehicle(scenario['vehicle'])
                self.setup_camera()
                
                # Wait for world to stabilize
                time.sleep(2.0)
                
                # Collect images for this scenario
                for i in range(images_per_scenario):
                    try:
                        image_filename, annotations = self.collect_frame_data(output_path)
                        
                        if image_filename and annotations:
                            # Create COCO-style annotation
                            image_info = {
                                'id': len(collected_images),
                                'file_name': image_filename,
                                'width': 1280,
                                'height': 720,
                                'scenario': scenario
                            }
                            collected_images.append(image_info)
                            
                            # Add annotations with image_id
                            for ann in annotations:
                                ann['id'] = len(collected_annotations)
                                ann['image_id'] = image_info['id']
                                collected_annotations.append(ann)
                            
                            if (i + 1) % 50 == 0:
                                print(f"  📸 Collected {i + 1}/{images_per_scenario} images")
                        
                        # Small delay between captures
                        time.sleep(0.1)
                        
                    except Exception as e:
                        print(f"  ⚠️  Error collecting frame {i}: {e}")
                        continue
                
            except Exception as e:
                print(f"❌ Scenario {scenario_idx + 1} failed: {e}")
                continue
            
            finally:
                # Cleanup scenario
                if self.camera:
                    self.camera.destroy()
                if self.vehicle:
                    self.vehicle.destroy()
                time.sleep(1.0)
        
        # Save COCO annotations
        coco_data = {
            'images': collected_images,
            'annotations': collected_annotations,
            'categories': [
                {'id': 1, 'name': 'Car'},
                {'id': 2, 'name': 'Bus'},
                {'id': 3, 'name': 'Person'},
                {'id': 4, 'name': 'Truck'},
                {'id': 5, 'name': 'Motorcycle'},
                {'id': 6, 'name': 'Bicycle'}
            ]
        }
        
        annotation_file = os.path.join(output_path, 'annotations', 'carla_annotations.json')
        with open(annotation_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"\n✅ Dataset collection complete:")
        print(f"  📸 Images: {len(collected_images)}")
        print(f"  🏷️  Annotations: {len(collected_annotations)}")
        print(f"  📁 Output: {output_path}")
    
    def cleanup(self):
        """Cleanup actors"""
        if self.camera:
            self.camera.destroy()
        if self.vehicle:
            self.vehicle.destroy()


def main():
    parser = argparse.ArgumentParser(description='CARLA Dataset Collection')
    
    # CARLA parameters
    parser.add_argument('--host', type=str, default='localhost',
                       help='CARLA server host (default: localhost)')
    parser.add_argument('--port', type=int, default=2000,
                       help='CARLA server port (default: 2000)')
    
    # Collection parameters
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for dataset')
    parser.add_argument('--num-images', type=int, default=1000,
                       help='Number of images to collect (default: 1000)')
    parser.add_argument('--scenarios', type=str, default=None,
                       help='JSON file with custom scenarios')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load scenarios if provided
    scenarios = None
    if args.scenarios:
        with open(args.scenarios, 'r') as f:
            scenarios = json.load(f)
    
    print("🚀 Starting CARLA dataset collection...")
    
    collector = CARLADataCollector(args.host, args.port)
    
    try:
        collector.collect_dataset(args.output, args.num_images, scenarios)
    except KeyboardInterrupt:
        print("\n⚠️  Collection interrupted by user")
    except Exception as e:
        print(f"\n❌ Collection failed: {e}")
        logging.exception("Collection error details:")
    finally:
        collector.cleanup()
        print("🔄 Cleanup complete")


if __name__ == '__main__':
    main()
