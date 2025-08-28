# CARLA SSD Integration Scripts

This directory contains all the scripts needed to run SSD object detection in the CARLA autonomous driving simulator.

## 📁 Directory Structure

```
scripts/
├── run_carla_detection.py      # Main detection script
├── run_carla_detection.bat     # Windows launcher for detection
├── benchmark_carla.py          # Performance benchmarking
├── benchmark_carla.bat         # Windows launcher for benchmarking
├── collect_carla_data.py       # Dataset collection from CARLA
├── collect_carla_data.bat      # Windows launcher for data collection
├── setup_carla.py              # CARLA setup and testing
├── setup_carla.bat             # Windows launcher for setup
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Setup CARLA Environment

First, test your CARLA installation and connection:

**Windows:**
```cmd
setup_carla.bat
```

**Python:**
```bash
python setup_carla.py --full-test
```

### 2. Run Real-time Detection

Run SSD object detection in CARLA:

**Windows:**
```cmd
run_carla_detection.bat
```

**Python:**
```bash
python run_carla_detection.py --checkpoint ../../experiments/synthetic/outputs/models/ssd300_final.pth
```

### 3. Benchmark Performance

Test model performance across different scenarios:

**Windows:**
```cmd
benchmark_carla.bat
```

**Python:**
```bash
python benchmark_carla.py --checkpoint ../../experiments/synthetic/outputs/models/ssd300_final.pth
```

### 4. Collect Training Data

Collect new training data from CARLA:

**Windows:**
```cmd
collect_carla_data.bat --output ../../data/new_carla_dataset
```

**Python:**
```bash
python collect_carla_data.py --output ../../data/new_carla_dataset --num-images 1000
```

## 📋 Prerequisites

### CARLA Installation

1. **Download CARLA**: Get CARLA 0.9.15 from [GitHub releases](https://github.com/carla-simulator/carla/releases)
2. **Extract CARLA**: Extract to desired location (e.g., `C:\CARLA_0.9.15`)
3. **Install Python API**:
   ```bash
   pip install carla
   ```
   Or from CARLA distribution:
   ```bash
   cd CARLA_ROOT/PythonAPI/carla/dist
   pip install carla-0.9.15-py3-none-any.whl
   ```

### System Requirements

- **GPU**: NVIDIA GTX 1060 or better (for real-time performance)
- **RAM**: 8GB+ recommended
- **Storage**: 5GB+ for CARLA installation
- **OS**: Windows 10/11 or Ubuntu 18.04+

### Python Dependencies

Install required packages:
```bash
pip install torch torchvision opencv-python numpy carla
```

## 🔧 Configuration

### CARLA Server

Start CARLA server before running scripts:

**Default (with graphics):**
```cmd
cd C:\CARLA_0.9.15
CarlaUE4.exe
```

**Headless (no graphics):**
```cmd
cd C:\CARLA_0.9.15
CarlaUE4.exe -RenderOffScreen
```

**Custom port:**
```cmd
CarlaUE4.exe -carla-rpc-port=2001
```

### Model Checkpoints

Ensure your trained SSD model is available:
- **Synthetic model**: `../../experiments/synthetic/outputs/models/ssd300_final.pth`
- **Real-world model**: `../../experiments/realworld/outputs/models/ssd300_final.pth`

## 📖 Script Details

### run_carla_detection.py

Main script for real-time object detection in CARLA.

**Features:**
- Real-time SSD inference on CARLA camera feed
- Configurable confidence thresholds
- Multiple camera positions (front, hood, roof)
- Weather and vehicle selection
- Video recording capability
- Headless operation support

**Usage:**
```bash
python run_carla_detection.py [options]

Options:
  --checkpoint PATH       Path to SSD model checkpoint (required)
  --confidence FLOAT      Detection confidence threshold (default: 0.3)
  --host HOST            CARLA server host (default: localhost)
  --port PORT            CARLA server port (default: 2000)
  --vehicle VEHICLE      Vehicle type to spawn
  --camera-position POS  Camera position: front, hood, roof
  --weather WEATHER      Weather preset
  --save-video           Save detection video
  --output PATH          Output video path
  --no-display          Run without display (headless)
  --max-frames N         Maximum frames to process
```

**Examples:**
```bash
# Basic detection
python run_carla_detection.py --checkpoint model.pth

# High confidence, save video
python run_carla_detection.py --checkpoint model.pth --confidence 0.5 --save-video

# Headless operation
python run_carla_detection.py --checkpoint model.pth --no-display --max-frames 1000
```

### benchmark_carla.py

Performance benchmarking across multiple scenarios.

**Features:**
- Automated testing across weather conditions
- FPS measurement and analysis
- Detection counting by object class
- JSON result export
- Multiple vehicle and camera configurations

**Usage:**
```bash
python benchmark_carla.py [options]

Options:
  --checkpoint PATH       Path to SSD model checkpoint (required)
  --duration SECONDS      Duration per scenario (default: 30)
  --scenarios FILE        JSON file with custom scenarios
  --output FILE           Output JSON file for results
```

**Default Scenarios:**
- Clear Day Highway
- Foggy City
- Night Driving
- Heavy Rain

### collect_carla_data.py

Collect annotated training data from CARLA.

**Features:**
- Automatic COCO format annotation generation
- Multiple weather and vehicle scenarios
- Configurable image collection count
- Bounding box projection from 3D to 2D
- Support for all 6 object classes

**Usage:**
```bash
python collect_carla_data.py [options]

Options:
  --output PATH           Output directory for dataset (required)
  --num-images N          Number of images to collect (default: 1000)
  --scenarios FILE        JSON file with custom scenarios
```

### setup_carla.py

CARLA environment setup and testing.

**Features:**
- CARLA installation verification
- Connection testing
- Basic functionality tests
- Vehicle spawn testing
- Camera setup verification

**Usage:**
```bash
python setup_carla.py [options]

Options:
  --setup-only           Only show setup instructions
  --connection-only      Only test connection
  --full-test           Run complete test suite
```

## 🎯 Controls and Interaction

### During Detection

- **'q'**: Quit detection
- **'s'**: Take screenshot
- **Ctrl+C**: Emergency stop

### Vehicle Control

- **Autopilot**: Enabled by default
- **Manual control**: Use `--no-autopilot` flag
- **Speed**: Automatically controlled by CARLA traffic manager

## 📊 Output Files

### Detection Videos

Saved to `../../outputs/carla/` with automatic timestamps:
```
carla_detection_ssd300_final_20240115_143022.mp4
```

### Benchmark Results

JSON files with detailed performance metrics:
```json
{
  "Clear_Day_Highway": {
    "avg_fps": 45.2,
    "total_detections": 234,
    "detections_per_class": {
      "Car": 198,
      "Person": 36
    }
  }
}
```

### Collected Datasets

COCO format with images and annotations:
```
dataset/
├── images/
│   ├── carla_image_001.png
│   └── ...
└── annotations/
    └── carla_annotations.json
```

## 🛠️ Troubleshooting

### Common Issues

**"Cannot connect to CARLA"**
- Ensure CARLA server is running
- Check host and port settings
- Verify firewall isn't blocking connection

**"Import carla could not be resolved"**
- Install CARLA Python API: `pip install carla`
- Check Python path configuration

**Low FPS performance**
- Use dedicated GPU
- Close unnecessary applications
- Consider headless mode for better performance
- Reduce image resolution in camera settings

**No detections appearing**
- Lower confidence threshold
- Check model checkpoint path
- Ensure objects are in camera view
- Verify model was trained on similar classes

### Performance Optimization

**For better FPS:**
```bash
# Use smaller images
python run_carla_detection.py --checkpoint model.pth --no-display

# Headless CARLA server
CarlaUE4.exe -RenderOffScreen

# Higher confidence threshold
python run_carla_detection.py --checkpoint model.pth --confidence 0.5
```

**For better detection accuracy:**
```bash
# Lower confidence threshold
python run_carla_detection.py --checkpoint model.pth --confidence 0.2

# Use hood camera for closer view
python run_carla_detection.py --checkpoint model.pth --camera-position hood
```

## 🔗 Integration with Training Pipeline

The CARLA scripts integrate seamlessly with the main training pipeline:

1. **Train model** using `src/training/train_ssd.py`
2. **Test in CARLA** using `scripts/run_carla_detection.py`
3. **Collect more data** using `scripts/collect_carla_data.py`
4. **Retrain** with combined datasets
5. **Benchmark performance** using `scripts/benchmark_carla.py`

## 📝 Custom Scenarios

Create custom benchmark scenarios in JSON format:

```json
[
  {
    "name": "Custom_Scenario",
    "weather": "ClearNoon",
    "vehicle": "vehicle.tesla.model3",
    "camera_position": "front",
    "autopilot": true,
    "confidence": 0.3
  }
]
```

Use with:
```bash
python benchmark_carla.py --scenarios custom_scenarios.json
```

## 📞 Support

For issues with:
- **CARLA installation**: Check [CARLA documentation](https://carla.readthedocs.io/)
- **Script errors**: Check log output and error messages
- **Performance issues**: Review troubleshooting section above
- **Integration problems**: Ensure all dependencies are installed

## 🎉 Success Indicators

You'll know everything is working correctly when:
- ✅ `setup_carla.py --full-test` passes all tests
- ✅ Real-time detection shows bounding boxes on vehicles and pedestrians
- ✅ FPS is above 20 for smooth operation
- ✅ Benchmark completes without errors
- ✅ Dataset collection generates COCO format annotations

Happy detecting! 🚗🎯
