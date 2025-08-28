#!/usr/bin/env python3
"""
CARLA Setup and Testing Script
Setup CARLA environment and run basic tests
"""

import argparse
import logging
import os
import sys
import time

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from carla_integration.utils.carla_utils import test_carla_connection


def test_carla_basic():
    """Test basic CARLA functionality"""
    print("🔧 Testing basic CARLA functionality...")
    
    try:
        import carla
        print("  ✅ CARLA Python API imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import CARLA Python API: {e}")
        print("     Please ensure CARLA Python API is installed:")
        print("     pip install carla")
        return False
    
    return True


def test_carla_world(host='localhost', port=2000):
    """Test CARLA world connection and basic operations"""
    print("🌍 Testing CARLA world operations...")
    
    try:
        import carla
        
        # Connect to CARLA
        client = carla.Client(host, port)
        client.set_timeout(10.0)
        
        # Get world
        world = client.get_world()
        print(f"  ✅ Connected to world: {world.get_map().name}")
        
        # Test weather
        weather = world.get_weather()
        print(f"  ✅ Current weather: cloudiness={weather.cloudiness}, precipitation={weather.precipitation}")
        
        # Test spawn points
        spawn_points = world.get_map().get_spawn_points()
        print(f"  ✅ Available spawn points: {len(spawn_points)}")
        
        # Test blueprint library
        blueprint_library = world.get_blueprint_library()
        vehicles = blueprint_library.filter('vehicle.*')
        cameras = blueprint_library.filter('sensor.camera.*')
        print(f"  ✅ Available vehicles: {len(vehicles)}")
        print(f"  ✅ Available cameras: {len(cameras)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ CARLA world test failed: {e}")
        return False


def test_vehicle_spawn(host='localhost', port=2000):
    """Test vehicle spawning and basic control"""
    print("🚗 Testing vehicle spawn and control...")
    
    try:
        import carla
        import random
        
        client = carla.Client(host, port)
        client.set_timeout(10.0)
        world = client.get_world()
        
        # Get vehicle blueprint
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        
        # Get spawn point
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        
        # Spawn vehicle
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(f"  ✅ Vehicle spawned at: {spawn_point.location}")
        
        # Test autopilot
        vehicle.set_autopilot(True)
        print("  ✅ Autopilot enabled")
        
        # Wait a bit
        time.sleep(2.0)
        
        # Get vehicle status
        location = vehicle.get_location()
        velocity = vehicle.get_velocity()
        speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
        print(f"  ✅ Vehicle location: ({location.x:.1f}, {location.y:.1f}, {location.z:.1f})")
        print(f"  ✅ Vehicle speed: {speed:.1f} m/s")
        
        # Cleanup
        vehicle.destroy()
        print("  ✅ Vehicle destroyed")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Vehicle test failed: {e}")
        return False


def test_camera_setup(host='localhost', port=2000):
    """Test camera setup and image capture"""
    print("📷 Testing camera setup and capture...")
    
    try:
        import carla
        import numpy as np
        import random
        
        client = carla.Client(host, port)
        client.set_timeout(10.0)
        world = client.get_world()
        
        # Spawn vehicle
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        
        # Setup camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '90')
        
        camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        print("  ✅ Camera attached to vehicle")
        
        # Capture image
        image_captured = [False]
        
        def image_callback(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            print(f"  ✅ Image captured: {image.width}x{image.height} pixels")
            image_captured[0] = True
        
        camera.listen(image_callback)
        
        # Wait for image
        start_time = time.time()
        while not image_captured[0] and time.time() - start_time < 5.0:
            time.sleep(0.1)
        
        if image_captured[0]:
            print("  ✅ Image callback successful")
        else:
            print("  ⚠️  No image received within timeout")
        
        # Cleanup
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        print("  ✅ Camera and vehicle destroyed")
        
        return image_captured[0]
        
    except Exception as e:
        print(f"  ❌ Camera test failed: {e}")
        return False


def run_full_test_suite(host='localhost', port=2000):
    """Run complete CARLA test suite"""
    print("🧪 Running CARLA Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic API Import", test_carla_basic),
        ("Connection Test", lambda: test_carla_connection(host, port)),
        ("World Operations", lambda: test_carla_world(host, port)),
        ("Vehicle Spawn", lambda: test_vehicle_spawn(host, port)),
        ("Camera Setup", lambda: test_camera_setup(host, port)),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔄 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {status}")
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! CARLA is ready for SSD detection.")
    else:
        print("⚠️  Some tests failed. Please check CARLA installation and setup.")
    
    return passed == total


def setup_carla_environment():
    """Setup and configure CARLA environment"""
    print("🔧 Setting up CARLA environment...")
    
    # Check if CARLA Python API is available
    try:
        import carla
        print("  ✅ CARLA Python API is available")
    except ImportError:
        print("  ❌ CARLA Python API not found")
        print("\nTo install CARLA Python API:")
        print("1. Download CARLA from: https://github.com/carla-simulator/carla/releases")
        print("2. Extract CARLA")
        print("3. Install Python API:")
        print("   cd CARLA_ROOT/PythonAPI/carla/dist")
        print("   pip install carla-X.X.X-py3-none-any.whl")
        print("\nOr install via pip:")
        print("   pip install carla")
        return False
    
    # Additional setup recommendations
    print("\n📋 CARLA Setup Recommendations:")
    print("1. Ensure CARLA server is running (CarlaUE4.exe)")
    print("2. Default port: 2000")
    print("3. For better performance:")
    print("   - Use dedicated GPU")
    print("   - Close unnecessary applications")
    print("   - Set CARLA to high performance mode")
    print("4. For headless operation:")
    print("   - Use -RenderOffScreen flag")
    print("   - Consider using Docker")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='CARLA Setup and Testing')
    
    # CARLA parameters
    parser.add_argument('--host', type=str, default='localhost',
                       help='CARLA server host (default: localhost)')
    parser.add_argument('--port', type=int, default=2000,
                       help='CARLA server port (default: 2000)')
    
    # Test options
    parser.add_argument('--setup-only', action='store_true',
                       help='Only show setup instructions')
    parser.add_argument('--connection-only', action='store_true',
                       help='Only test connection')
    parser.add_argument('--full-test', action='store_true',
                       help='Run full test suite')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if args.setup_only:
        setup_carla_environment()
    elif args.connection_only:
        print("🔄 Testing CARLA connection...")
        if test_carla_connection(args.host, args.port):
            print("✅ Connection successful!")
        else:
            print("❌ Connection failed!")
    elif args.full_test:
        run_full_test_suite(args.host, args.port)
    else:
        # Default: setup and basic test
        if setup_carla_environment():
            print("\n🔄 Testing basic connection...")
            test_carla_connection(args.host, args.port)


if __name__ == '__main__':
    main()
