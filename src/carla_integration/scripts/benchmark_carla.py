#!/usr/bin/env python3
"""
CARLA Performance Benchmark Script
Evaluate SSD model performance in various CARLA scenarios
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from carla_integration.core.detection_system import CARLADetectionSystem
from carla_integration.utils.carla_utils import test_carla_connection


class CARLABenchmark:
    def __init__(self, checkpoint_path, carla_host='localhost', carla_port=2000):
        self.checkpoint_path = checkpoint_path
        self.carla_host = carla_host
        self.carla_port = carla_port
        self.results = defaultdict(list)
        
    def run_scenario_benchmark(self, scenario_config, duration=30):
        """Run benchmark for a specific scenario"""
        print(f"🔄 Running scenario: {scenario_config['name']}")
        
        # Initialize detection system
        detection_system = CARLADetectionSystem(
            checkpoint_path=self.checkpoint_path,
            confidence_threshold=scenario_config.get('confidence', 0.3),
            carla_host=self.carla_host,
            carla_port=self.carla_port
        )
        
        try:
            # Setup environment
            detection_system.setup(
                vehicle_filter=scenario_config.get('vehicle', 'vehicle.tesla.model3'),
                camera_position=scenario_config.get('camera_position', 'front'),
                weather=scenario_config.get('weather', 'ClearNoon'),
                autopilot=scenario_config.get('autopilot', True)
            )
            
            # Run benchmark
            start_time = time.time()
            frame_count = 0
            detection_counts = defaultdict(int)
            fps_samples = []
            
            print(f"  Running for {duration} seconds...")
            
            while time.time() - start_time < duration:
                frame_start = time.time()
                
                # Process one frame
                try:
                    image, detections = detection_system.process_frame()
                    frame_count += 1
                    
                    # Count detections by class
                    for detection in detections:
                        class_name = detection.get('class_name', 'Unknown')
                        detection_counts[class_name] += 1
                    
                    # Calculate FPS
                    frame_time = time.time() - frame_start
                    if frame_time > 0:
                        fps_samples.append(1.0 / frame_time)
                    
                except Exception as e:
                    print(f"    Frame processing error: {e}")
                    continue
            
            # Calculate metrics
            total_time = time.time() - start_time
            avg_fps = sum(fps_samples) / len(fps_samples) if fps_samples else 0
            total_detections = sum(detection_counts.values())
            
            scenario_results = {
                'scenario': scenario_config['name'],
                'duration': total_time,
                'frames': frame_count,
                'avg_fps': avg_fps,
                'total_detections': total_detections,
                'detections_per_class': dict(detection_counts),
                'config': scenario_config
            }
            
            self.results[scenario_config['name']] = scenario_results
            
            print(f"  ✅ Completed: {frame_count} frames, {avg_fps:.1f} FPS, {total_detections} detections")
            
        finally:
            detection_system.cleanup()
    
    def save_results(self, output_path):
        """Save benchmark results to JSON"""
        with open(output_path, 'w') as f:
            json.dump(dict(self.results), f, indent=2)
        print(f"📊 Results saved to: {output_path}")
    
    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "="*60)
        print("CARLA BENCHMARK SUMMARY")
        print("="*60)
        
        for scenario_name, results in self.results.items():
            print(f"\n📋 {scenario_name}:")
            print(f"  FPS: {results['avg_fps']:.1f}")
            print(f"  Frames: {results['frames']}")
            print(f"  Total Detections: {results['total_detections']}")
            
            if results['detections_per_class']:
                print("  Detections by class:")
                for class_name, count in results['detections_per_class'].items():
                    print(f"    {class_name}: {count}")


def get_default_scenarios():
    """Get default benchmark scenarios"""
    return [
        {
            'name': 'Clear_Day_Highway',
            'weather': 'ClearNoon',
            'vehicle': 'vehicle.tesla.model3',
            'camera_position': 'front',
            'autopilot': True,
            'confidence': 0.3
        },
        {
            'name': 'Foggy_City',
            'weather': 'SoftRainSunset',
            'vehicle': 'vehicle.audi.etron',
            'camera_position': 'front',
            'autopilot': True,
            'confidence': 0.25
        },
        {
            'name': 'Night_Driving',
            'weather': 'ClearSunset',
            'vehicle': 'vehicle.bmw.grandtourer',
            'camera_position': 'front',
            'autopilot': True,
            'confidence': 0.35
        },
        {
            'name': 'Heavy_Rain',
            'weather': 'HardRainNoon',
            'vehicle': 'vehicle.tesla.model3',
            'camera_position': 'hood',
            'autopilot': True,
            'confidence': 0.4
        }
    ]


def main():
    parser = argparse.ArgumentParser(description='CARLA SSD Benchmark')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to SSD model checkpoint')
    
    # CARLA parameters
    parser.add_argument('--host', type=str, default='localhost',
                       help='CARLA server host (default: localhost)')
    parser.add_argument('--port', type=int, default=2000,
                       help='CARLA server port (default: 2000)')
    
    # Benchmark parameters
    parser.add_argument('--duration', type=int, default=30,
                       help='Duration per scenario in seconds (default: 30)')
    parser.add_argument('--scenarios', type=str, default=None,
                       help='JSON file with custom scenarios')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Validate checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    # Test CARLA connection
    print("🔄 Testing CARLA connection...")
    if not test_carla_connection(args.host, args.port):
        print("❌ Cannot connect to CARLA")
        return
    
    # Load scenarios
    if args.scenarios:
        with open(args.scenarios, 'r') as f:
            scenarios = json.load(f)
    else:
        scenarios = get_default_scenarios()
    
    print(f"🚀 Starting benchmark with {len(scenarios)} scenarios")
    
    # Run benchmark
    benchmark = CARLABenchmark(args.checkpoint, args.host, args.port)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}]", end=" ")
        try:
            benchmark.run_scenario_benchmark(scenario, args.duration)
        except KeyboardInterrupt:
            print("\n⚠️  Benchmark interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Scenario failed: {e}")
            continue
    
    # Save and display results
    if args.output:
        benchmark.save_results(args.output)
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_output = f"../../outputs/carla/benchmark_results_{timestamp}.json"
        os.makedirs(os.path.dirname(default_output), exist_ok=True)
        benchmark.save_results(default_output)
    
    benchmark.print_summary()


if __name__ == '__main__':
    main()
