#!/usr/bin/env python3
"""
Main CARLA SSD Detection Script
Entry point for running SSD object detection in CARLA simulator
"""

import argparse
import logging
import os
import sys

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from carla_integration.core.detection_system import CARLADetectionSystem
from carla_integration.utils.carla_utils import test_carla_connection


def main():
    parser = argparse.ArgumentParser(description='CARLA SSD Detection System')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to SSD model checkpoint')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Detection confidence threshold (default: 0.3)')
    
    # CARLA parameters
    parser.add_argument('--host', type=str, default='localhost',
                       help='CARLA server host (default: localhost)')
    parser.add_argument('--port', type=int, default=2000,
                       help='CARLA server port (default: 2000)')
    parser.add_argument('--vehicle', type=str, default='vehicle.tesla.model3',
                       help='Vehicle type to spawn (default: vehicle.tesla.model3)')
    parser.add_argument('--camera-position', type=str, default='front',
                       choices=['front', 'hood', 'roof'],
                       help='Camera attachment position (default: front)')
    parser.add_argument('--weather', type=str, default='ClearNoon',
                       help='Weather preset (default: ClearNoon)')
    parser.add_argument('--no-autopilot', action='store_true',
                       help='Disable vehicle autopilot')
    
    # Output parameters
    parser.add_argument('--save-video', action='store_true',
                       help='Save detection video')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (auto-generated if not specified)')
    parser.add_argument('--no-display', action='store_true',
                       help='Run without display (headless mode)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce detection logging output')
    
    # Testing parameters
    parser.add_argument('--test-connection', action='store_true',
                       help='Test CARLA connection and exit')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Test connection if requested
    if args.test_connection:
        print("Testing CARLA connection...")
        test_carla_connection(args.host, args.port)
        return
    
    # Validate checkpoint path
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint file not found: {args.checkpoint}")
        print("\nAvailable checkpoint locations:")
        print("  - experiments/synthetic/outputs/models/ssd300_final.pth")
        print("  - experiments/realworld/outputs/models/ssd300_final.pth")
        return
    
    # Test CARLA connection before starting
    print("🔄 Testing CARLA connection...")
    if not test_carla_connection(args.host, args.port):
        print("\n❌ Cannot connect to CARLA. Please ensure:")
        print("1. CARLA simulator is running")
        print("2. CARLA is accessible at the specified host and port")
        print("3. No firewall is blocking the connection")
        return
    
    print("\n🚀 Starting CARLA SSD Detection System...")
    
    # Create detection system
    try:
        detection_system = CARLADetectionSystem(
            checkpoint_path=args.checkpoint,
            confidence_threshold=args.confidence,
            carla_host=args.host,
            carla_port=args.port
        )
    except Exception as e:
        print(f"❌ Failed to initialize detection system: {e}")
        return
    
    try:
        # Setup CARLA environment
        print("🔧 Setting up CARLA environment...")
        detection_system.setup(
            vehicle_filter=args.vehicle,
            camera_position=args.camera_position,
            weather=args.weather,
            autopilot=not args.no_autopilot
        )
        
        # Setup output path
        output_path = args.output
        if args.save_video and output_path is None:
            # Auto-generate output path
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = os.path.basename(args.checkpoint).replace('.pth', '')
            output_path = f"../../outputs/carla/carla_detection_{model_name}_{timestamp}.mp4"
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Run detection
        print("🎯 Starting detection loop...")
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to take screenshot")
        print("  - Press Ctrl+C to emergency stop")
        
        detection_system.run_detection_loop(
            display=not args.no_display,
            save_video=args.save_video,
            output_path=output_path,
            max_frames=args.max_frames,
            print_detections=not args.quiet
        )
        
    except KeyboardInterrupt:
        print("\n⚠️  Detection interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during detection: {e}")
        logging.exception("Detection error details:")
    
    finally:
        # Cleanup
        print("🔄 Cleaning up...")
        detection_system.cleanup()
        print("✅ Detection system shutdown complete")


if __name__ == '__main__':
    main()
