"""
CARLA Environment Management
Handles CARLA client connection, vehicle spawning, and sensor setup
"""

import carla
import numpy as np
import cv2
import time
import logging
import queue
import random


class CARLAEnvironment:
    """CARLA Environment Manager"""
    
    def __init__(self, host='localhost', port=2000, timeout=10.0):
        """Initialize CARLA environment"""
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera_sensor = None
        self.image_queue = queue.Queue(maxsize=10)
        self.host = host
        self.port = port
        
        try:
            # Connect to CARLA
            self.client = carla.Client(host, port)
            self.client.set_timeout(timeout)
            self.world = self.client.get_world()
            
            # Get world info
            self.map = self.world.get_map()
            self.blueprint_library = self.world.get_blueprint_library()
            
            logging.info(f"Connected to CARLA server at {host}:{port}")
            logging.info(f"Map: {self.map.name}")
            
        except Exception as e:
            logging.error(f"Failed to connect to CARLA: {e}")
            raise
    
    def list_available_vehicles(self):
        """List all available vehicle blueprints"""
        vehicles = self.blueprint_library.filter('vehicle.*')
        vehicle_list = [bp.id for bp in vehicles]
        logging.info(f"Available vehicles ({len(vehicle_list)}): {vehicle_list[:10]}...")  # Show first 10
        return vehicle_list
    
    def spawn_vehicle(self, vehicle_filter='vehicle.tesla.model3'):
        """Spawn a vehicle in the world"""
        try:
            vehicle_bp = self.blueprint_library.filter(vehicle_filter)[0]
        except IndexError:
            logging.warning(f"Vehicle {vehicle_filter} not found, using random vehicle")
            vehicles = self.blueprint_library.filter('vehicle.*')
            vehicle_bp = random.choice(list(vehicles))
        
        # Get spawn points
        spawn_points = self.map.get_spawn_points()
        if not spawn_points:
            raise Exception("No spawn points available on this map")
        
        # Try multiple spawn points if first one fails
        for i, spawn_point in enumerate(spawn_points[:10]):  # Try first 10 spawn points
            try:
                self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                logging.info(f"Spawned vehicle: {vehicle_bp.id} at spawn point {i}")
                break
            except Exception as e:
                logging.warning(f"Failed to spawn at point {i}: {e}")
                continue
        else:
            raise Exception("Failed to spawn vehicle at any spawn point")
        
        return self.vehicle
    
    def setup_camera(self, width=800, height=600, fov=90, attachment_type='front'):
        """Setup RGB camera sensor"""
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        
        # Set camera attributes
        camera_bp.set_attribute('image_size_x', str(width))
        camera_bp.set_attribute('image_size_y', str(height))
        camera_bp.set_attribute('fov', str(fov))
        
        # Camera attachment positions
        camera_transforms = {
            'front': carla.Transform(
                carla.Location(x=1.5, z=2.4),  # Front bumper, 2.4m high
                carla.Rotation(pitch=0.0)
            ),
            'hood': carla.Transform(
                carla.Location(x=0.8, z=1.7),  # Hood
                carla.Rotation(pitch=0.0)
            ),
            'roof': carla.Transform(
                carla.Location(x=0.0, z=2.8),  # Roof center
                carla.Rotation(pitch=0.0)
            )
        }
        
        camera_transform = camera_transforms.get(attachment_type, camera_transforms['front'])
        
        # Spawn camera
        self.camera_sensor = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle
        )
        
        # Start listening
        self.camera_sensor.listen(self._on_camera_data)
        logging.info(f"Camera sensor setup complete ({attachment_type} attachment)")
        
        return self.camera_sensor
    
    def _on_camera_data(self, image):
        """Camera data callback"""
        # Convert CARLA image to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))  # RGBA
        array = array[:, :, :3]  # RGB only
        array = array[:, :, ::-1]  # BGR to RGB for OpenCV
        
        # Add to queue (remove oldest if queue is full)
        if self.image_queue.full():
            try:
                self.image_queue.get_nowait()
            except queue.Empty:
                pass
        
        self.image_queue.put(array)
    
    def get_latest_image(self):
        """Get the latest camera image"""
        try:
            return self.image_queue.get_nowait()
        except queue.Empty:
            return None
    
    def set_vehicle_autopilot(self, enabled=True):
        """Enable/disable vehicle autopilot"""
        if self.vehicle:
            self.vehicle.set_autopilot(enabled)
            logging.info(f"Autopilot {'enabled' if enabled else 'disabled'}")
    
    def set_weather(self, weather_preset='ClearNoon'):
        """Set weather conditions"""
        weather_presets = {
            'ClearNoon': carla.WeatherParameters.ClearNoon,
            'CloudyNoon': carla.WeatherParameters.CloudyNoon,
            'WetNoon': carla.WeatherParameters.WetNoon,
            'WetCloudyNoon': carla.WeatherParameters.WetCloudyNoon,
            'MidRainyNoon': carla.WeatherParameters.MidRainyNoon,
            'HardRainNoon': carla.WeatherParameters.HardRainNoon,
            'SoftRainNoon': carla.WeatherParameters.SoftRainNoon,
            'ClearSunset': carla.WeatherParameters.ClearSunset,
            'CloudySunset': carla.WeatherParameters.CloudySunset,
            'WetSunset': carla.WeatherParameters.WetSunset,
            'WetCloudySunset': carla.WeatherParameters.WetCloudySunset,
            'MidRainSunset': carla.WeatherParameters.MidRainSunset,
            'HardRainSunset': carla.WeatherParameters.HardRainSunset,
            'SoftRainSunset': carla.WeatherParameters.SoftRainSunset,
        }
        
        weather = weather_presets.get(weather_preset, carla.WeatherParameters.ClearNoon)
        self.world.set_weather(weather)
        logging.info(f"Weather set to: {weather_preset}")
    
    def get_vehicle_location(self):
        """Get current vehicle location"""
        if self.vehicle:
            location = self.vehicle.get_location()
            return (location.x, location.y, location.z)
        return None
    
    def get_vehicle_speed(self):
        """Get current vehicle speed in km/h"""
        if self.vehicle:
            velocity = self.vehicle.get_velocity()
            speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            return speed
        return 0.0
    
    def wait_for_first_image(self, timeout=10.0):
        """Wait for first camera image"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.get_latest_image() is not None:
                return True
            time.sleep(0.1)
        return False
    
    def cleanup(self):
        """Clean up CARLA actors"""
        if self.camera_sensor:
            self.camera_sensor.destroy()
            logging.info("Camera sensor destroyed")
        
        if self.vehicle:
            self.vehicle.destroy()
            logging.info("Vehicle destroyed")
        
        logging.info("CARLA cleanup complete")
