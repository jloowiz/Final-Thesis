"""
CARLA Utilities
Helper functions for CARLA integration
"""

import carla
import logging
import time


def test_carla_connection(host='localhost', port=2000, timeout=10.0):
    """Test CARLA connection and display system info"""
    
    try:
        print("🔄 Connecting to CARLA...")
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        
        # Get world
        world = client.get_world()
        
        # Get system info
        map_name = world.get_map().name
        weather = world.get_weather()
        
        print("✅ Connected to CARLA successfully!")
        print(f"📍 Map: {map_name}")
        print(f"🌤️  Weather: {weather}")
        
        # Get available blueprints
        blueprint_library = world.get_blueprint_library()
        vehicles = list(blueprint_library.filter('vehicle.*'))
        sensors = list(blueprint_library.filter('sensor.*'))
        
        print(f"🚗 Available vehicles: {len(vehicles)}")
        print(f"📷 Available sensors: {len(sensors)}")
        
        # List some popular vehicles
        popular_vehicles = [
            'vehicle.tesla.model3',
            'vehicle.bmw.grandtourer',
            'vehicle.audi.a2',
            'vehicle.mercedes.coupe',
            'vehicle.mustang.mustang'
        ]
        
        print("\n🎯 Popular vehicles to test:")
        for vehicle in popular_vehicles:
            try:
                bp = blueprint_library.filter(vehicle)[0]
                print(f"  ✅ {vehicle}")
            except IndexError:
                print(f"  ❌ {vehicle} (not available)")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to CARLA: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure CARLA simulator is running")
        print("2. Check if CARLA is running on the correct port (default: 2000)")
        print("3. Verify CARLA installation")
        print("4. Try restarting CARLA simulator")
        return False


def list_carla_maps(host='localhost', port=2000):
    """List all available CARLA maps"""
    
    try:
        client = carla.Client(host, port)
        client.set_timeout(10.0)
        
        available_maps = client.get_available_maps()
        
        print(f"📍 Available CARLA Maps ({len(available_maps)}):")
        for i, map_name in enumerate(available_maps, 1):
            print(f"  {i}. {map_name}")
        
        return available_maps
        
    except Exception as e:
        print(f"❌ Failed to get maps: {e}")
        return []


def change_carla_map(map_name, host='localhost', port=2000):
    """Change CARLA map"""
    
    try:
        client = carla.Client(host, port)
        client.set_timeout(10.0)
        
        print(f"🔄 Loading map: {map_name}")
        world = client.load_world(map_name)
        
        print(f"✅ Map changed to: {map_name}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to change map: {e}")
        return False


def setup_carla_weather(weather_preset='ClearNoon', host='localhost', port=2000):
    """Setup CARLA weather"""
    
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
    
    try:
        client = carla.Client(host, port)
        client.set_timeout(10.0)
        world = client.get_world()
        
        weather = weather_presets.get(weather_preset)
        if weather is None:
            print(f"❌ Unknown weather preset: {weather_preset}")
            print(f"Available presets: {list(weather_presets.keys())}")
            return False
        
        world.set_weather(weather)
        print(f"🌤️  Weather set to: {weather_preset}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to set weather: {e}")
        return False


def get_carla_system_info(host='localhost', port=2000):
    """Get comprehensive CARLA system information"""
    
    try:
        client = carla.Client(host, port)
        client.set_timeout(10.0)
        world = client.get_world()
        
        # Basic info
        map_name = world.get_map().name
        weather = world.get_weather()
        settings = world.get_settings()
        
        print("🎮 CARLA System Information")
        print("="*40)
        print(f"Map: {map_name}")
        print(f"Weather: {weather}")
        print(f"Synchronous Mode: {settings.synchronous_mode}")
        print(f"Fixed Delta: {settings.fixed_delta_seconds}")
        print(f"No Rendering: {settings.no_rendering_mode}")
        
        # Actors info
        actors = world.get_actors()
        vehicles = actors.filter('vehicle.*')
        pedestrians = actors.filter('walker.pedestrian.*')
        sensors = actors.filter('sensor.*')
        
        print(f"\nActive Actors:")
        print(f"  Vehicles: {len(vehicles)}")
        print(f"  Pedestrians: {len(pedestrians)}")
        print(f"  Sensors: {len(sensors)}")
        print(f"  Total: {len(actors)}")
        
        # Available blueprints
        blueprint_library = world.get_blueprint_library()
        all_vehicles = list(blueprint_library.filter('vehicle.*'))
        all_sensors = list(blueprint_library.filter('sensor.*'))
        all_pedestrians = list(blueprint_library.filter('walker.pedestrian.*'))
        
        print(f"\nAvailable Blueprints:")
        print(f"  Vehicles: {len(all_vehicles)}")
        print(f"  Sensors: {len(all_sensors)}")
        print(f"  Pedestrians: {len(all_pedestrians)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to get system info: {e}")
        return False


if __name__ == "__main__":
    # Quick test when run directly
    test_carla_connection()
