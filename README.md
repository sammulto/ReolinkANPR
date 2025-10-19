# ReolinkANPR

Automatic Number Plate Recognition (ANPR) system for Reolink cameras with AI vehicle detection **and vehicle recognition**.

---

## What's New in v2.0

### Vehicle Recognition (October 2025)
- **Vehicle Color Detection** - Automatically detects vehicle colors using computer vision
- **Make & Model Recognition** - Identifies vehicle manufacturer and model using deep learning
- **Vehicle Detection & Cropping** - YOLOv9-based detection saves cropped images of vehicles
- **Multi-Vehicle Support** - Detects and selects the primary vehicle from multiple vehicles in frame
- **Vehicle-Only Tracking** - Tracks vehicles even when no plate is detected
- **Enhanced Notifications** - Telegram & Home Assistant now include full vehicle information
- **Smart Deduplication** - Separate 30s cooldown for plate vs. vehicle-based detections
- **Database Migration** - Automatically upgrades existing databases with new columns
- **Configurable** - Easy toggle in web UI, no code changes needed

**Upgrade Notes:**
- Existing databases auto-migrate on startup
- PyTorch required: `pip install torch torchvision`
- YOLOv9 required: `pip install ultralytics` (uses same YOLO architecture as ALPR)
- Falls back to center-crop estimation if YOLOv9 unavailable
- All features disabled by default for minimal impact
- See [VEHICLE_RECOGNITION.md](VEHICLE_RECOGNITION.md) for details

---

## Screenshots

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="screenshots/Dashboard.png" alt="Dashboard" width="800"/>
        <br/>
        <b>Dashboard</b> - Real-time detections with plate crops
      </td>
    </tr>
    <tr>
      <td align="center">
        <img src="screenshots/config.png" alt="Configuration" width="800"/>
        <br/>
        <b>Configuration</b> - Camera & ALPR settings
      </td>
    </tr>
    <tr>
      <td align="center">
        <img src="screenshots/config2.png" alt="Advanced Settings" width="800"/>
        <br/>
        <b>Advanced Settings</b> - Recording & notification options
      </td>
    </tr>
  </table>
</div>

## Vehicle Recognition Features

ReolinkANPR now includes advanced vehicle recognition capabilities powered by deep learning:

### What It Detects
- **Vehicle Color** - Black, White, Silver, Gray, Red, Blue, Green, Yellow, Orange, Brown, Gold, Cyan, Purple, Pink
  - Uses advanced HSV histogram analysis with shadow/reflection filtering
  - Multi-region sampling for accurate color detection
- **Vehicle Make & Model** - Powered by Stanford Cars/CompCars dataset
  - **Quick Start**: Run `python -m src.download_model` to get a real pre-trained model!
  - Supports 196+ vehicle classes (make/model/year)
  - See [QUICK_START_VEHICLE_MODEL.md](QUICK_START_VEHICLE_MODEL.md) for 2-minute setup
  - See [COMPCARS_SETUP.md](COMPCARS_SETUP.md) for advanced training/customization
- **Vehicle Location** - YOLOv9 detects and crops individual vehicles from multi-vehicle scenes

### Key Features
```markdown
# ReolinkANPR

Lightweight Automatic Number Plate Recognition (ANPR) for Reolink cameras.

This repository provides an event-driven service that listens to Reolink AI events, records short clips, runs plate OCR, and (optionally) performs basic vehicle analysis. Over time the project accumulated legacy and duplicated documentation; this README is a concise, up-to-date entry point. For detailed feature-specific docs see the files in the repository root (VEHICLE_RECOGNITION.md, NOTIFICATIONS.md, etc.).

## What this README contains
- Short overview and supported features
- Quick start (minimal steps to run)
- Where to find more detailed docs

## Supported features (high-level)
- TCP push event listener for Reolink cameras
- Short clip recording and frame extraction (ffmpeg)
- License plate detection and OCR
- Optional vehicle analysis (color, make/model) when models are present
- Web dashboard (Flask) for configuration and viewing detections
- Notification hooks for Telegram and Home Assistant

## Quick start (minimal)
Prerequisites: Python 3.8+ (3.11 recommended), ffmpeg. Optional: PyTorch for vehicle recognition.

1) Create and activate a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Copy the example config and edit it: `cp config.yaml.example config.yaml`

3) Run the service:

```bash
python run.py
```

4) Open the web UI: http://localhost:5001 (default)

For configuration options and advanced guides, use the in-app documentation (http://localhost:5001/docs) or the topic-specific markdown files in the repository root.

## Where to find detailed docs
- Vehicle recognition and models: VEHICLE_RECOGNITION.md, VEHICLE_MODEL_UPGRADE.md
- Notifications and integrations: NOTIFICATIONS.md
- Vehicle-only detections: VEHICLE_ONLY_DETECTION.md
- Development notes and tests: CONtributing and test files in the repo

## Contributing
Please open issues or pull requests. Keep changes focused and include tests where applicable.

## License
MIT (see LICENSE)

```
- **Vehicle recognition setup guide**

## How It Works

1. **Event Detection** - Camera AI detects vehicle → TCP push event
2. **Recording** - System records 5-6 seconds of RTSP stream
3. **Optimization** (Optional) - Applies sharp imaging settings for plate capture
4. **Frame Extraction** - Extracts all frames from recording
5. **ALPR Processing** - FastALPR detects and reads license plates
6. **Vehicle Recognition** - Analyzes vehicle for color, make, and model (even if no plate found)
7. **Deduplication** - Checks for duplicates (plate-based or vehicle-based for no-plate detections)
8. **Database Storage** - Saves plate number, vehicle attributes, confidence, images
9. **Notifications** - Sends to Telegram/Home Assistant with full vehicle info
10. **Settings Restore** (Optional) - Restores normal camera settings

### Vehicle Recognition Details

**Color Detection:**
- Uses HSV color space analysis
- Detects: Black, White, Silver, Gray, Red, Blue, Green, Yellow, Orange, Brown, etc.
- Very reliable (~95% accuracy)

**Make & Model Detection:**
- Uses deep learning with PyTorch/ResNet50
- Current implementation is proof-of-concept
- For production: See [VEHICLE_MODEL_UPGRADE.md](VEHICLE_MODEL_UPGRADE.md) for improvement options
  - Fine-tune on Stanford Cars dataset
  - Use commercial APIs (Google Vision, AWS Rekognition)
  - Train custom models

**Vehicle-Only Detections:**
- Saves vehicles even when no plate is detected
- Useful for traffic monitoring, parking enforcement, security
- Deduplication based on color + make combination
- Marked as "NO_PLATE" in database

## Architecture

```
┌─────────────────┐
│ Reolink Camera  │ (AI Vehicle Detection)
└────────┬────────┘
         │ TCP Push Event (Baichuan)
         ↓
┌─────────────────┐
│ Camera Client   │ (reolink_aio)
└────────┬────────┘
         │ Async
         ↓
┌─────────────────┐
│  ANPR Service   │ (Orchestration)
└────────┬────────┘
         │
    ┌────┴────┬────────┬──────────┬─────────┐
    ↓         ↓        ↓          ↓         ↓
┌──────┐ ┌──────┐ ┌────────┐ ┌────────┐ ┌─────┐
│ ALPR │ │Vehic.│ │   DB   │ │  Web   │ │ Log │
│      │ │Recog.│ │        │ │        │ │     │
└──────┘ └──────┘ └────────┘ └────────┘ └─────┘
FastALPR  PyTorch  SQLite     Flask
          ResNet50 (vehicle
                    attrs)
```

## Troubleshooting

### Camera Won't Connect

```bash
# Test camera connectivity
ping 192.168.1.100

# Check if camera is accessible
curl http://192.168.1.100
```

**RTSP Streaming:**
- RTSP is **automatically enabled** when the system connects
- If RTSP is disabled, the system will enable it on port 554
- Check logs for "RTSP enabled successfully" confirmation

### No Detections

1. **Check AI is enabled** on camera (Settings → AI → Vehicle Detection)
2. **Verify camera triggers** - wave at camera and check logs
3. **Check logs:** `tail -f logs/anpr.log`
4. **Vehicle recognition issues:**
   - Check if PyTorch is installed: `pip list | grep torch`
   - Look for "Vehicle recognition enabled" in logs
   - Disable if not needed: Config → ALPR → Uncheck "Vehicle Recognition"

### Low Vehicle Recognition Accuracy

**Color detection** is very reliable. For **make/model improvements:**
1. See [VEHICLE_MODEL_UPGRADE.md](VEHICLE_MODEL_UPGRADE.md) for enhancement options
2. Consider using commercial APIs for production
3. Fine-tune models on vehicle-specific datasets
4. Current implementation uses general ImageNet features (proof of concept)

### Settings Not Persisting

Check the in-app documentation at `http://localhost:5001/docs` for camera setup guides and troubleshooting tips.

## Development

### Project Structure

```
ReolinkANPR/
├── src/                    # Core application code
│   ├── anpr_service.py    # Main orchestration
│   ├── camera_client.py   # Reolink camera interface
│   ├── alpr_processor.py  # FastALPR integration
│   ├── vehicle_recognizer.py  # Vehicle recognition (color/make/model)
│   ├── database.py        # SQLite operations
│   ├── notifier.py        # Telegram & Home Assistant (enhanced)
│   └── logger.py          # Logging configuration
├── web/                    # Web dashboard
│   ├── app.py             # Flask application
│   ├── templates/         # HTML templates (includes docs)
│   └── static/            # CSS, JS, images
├── data/                   # Database and images
├── logs/                   # Application logs
├── config.yaml            # Configuration (gitignored)
├── config.yaml.example    # Example configuration
├── requirements.txt       # Python dependencies
├── VEHICLE_RECOGNITION.md # Vehicle recognition docs
├── VEHICLE_ONLY_DETECTION.md  # Vehicle-only tracking docs
├── VEHICLE_MODEL_UPGRADE.md   # Upgrade guide
└── NOTIFICATIONS.md       # Enhanced notifications guide
```

### Running Tests

```bash
# Test dependencies
python test_dependencies.py

# Check logs
tail -f logs/anpr.log

# Test camera settings via Web UI
# Go to http://localhost:5001/config → Camera Control Settings
```

## Documentation

All documentation is now built into the web interface:
- **Installation:** Follow the Quick Start guide above
- **Configuration:** Use the web UI at `http://localhost:5001/config`
- **Settings Guide:** Visit `http://localhost:5001/docs` for complete documentation
- **Troubleshooting:** Check the in-app docs and `logs/anpr.log`

**New Feature Documentation:**
- **[VEHICLE_RECOGNITION.md](VEHICLE_RECOGNITION.md)** - Complete vehicle recognition guide
- **[VEHICLE_ONLY_DETECTION.md](VEHICLE_ONLY_DETECTION.md)** - Track vehicles without plates
- **[VEHICLE_MODEL_UPGRADE.md](VEHICLE_MODEL_UPGRADE.md)** - Improve make/model accuracy
- **[NOTIFICATIONS.md](NOTIFICATIONS.md)** - Enhanced notifications with vehicle data

## Requirements

- **Python:** 3.11 or higher (for TCP push event support)
- **Operating System:** Linux, macOS, or Windows
- **Camera:** Reolink camera with AI vehicle detection
- **Network:** Local network access to camera
- **Optional:** PyTorch for vehicle recognition (auto-detects GPU/CPU)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Use Cases

### Smart Home Automation
- **Auto-open garage** when your car arrives (via Home Assistant)
- **Turn on lights** when vehicle detected at night
- **Voice announcements** ("John is home!")
- **Different actions** for known vs unknown plates
- **Color-based automation** - Different lights for red vs. blue cars
- **Make-based actions** - Special greeting for Tesla owners

### Security & Monitoring
- **Real-time alerts** for unexpected visitors
- **Log all vehicles** entering your property (even without readable plates)
- **Time-stamped records** with plate images and vehicle descriptions
- **Search history** by plate number or date
- **Vehicle tracking** - Monitor all traffic, not just plates
- **Unknown vehicle alerts** - Get notified of vehicles without plates

### Traffic Analysis
- **Vehicle statistics** - Track vehicle colors, makes, models
- **Traffic patterns** - Count all vehicles passing by
- **Parking monitoring** - Detect vehicles even if plates are obscured
- **Fleet management** - Track company vehicles by color/make

### Simple Setup
- Web UI for all configuration (no YAML editing required)
- Built-in documentation and examples
- 30-second deduplication prevents spam
- Dark/light theme for comfortable viewing
- One-click vehicle recognition toggle

## Acknowledgments

- [FastALPR](https://github.com/ankandrew/fast-alpr) - License plate recognition
- [reolink-aio](https://github.com/starkillerOG/reolink_aio) - Reolink camera API
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [PyTorch](https://pytorch.org/) - Deep learning framework for vehicle recognition
- [TorchVision](https://pytorch.org/vision/stable/index.html) - Computer vision models
- Originally created by **Harky** - This repo is forked from [https://github.com/Harky911/ReolinkANPR](https://github.com/Harky911/ReolinkANPR)

## Support

- **Issues:** [GitHub Issues](https://github.com/Harky911/ReolinkANPR/issues)
- **Documentation:** Built-in at `http://localhost:5001/docs`
- **Discussions:** [GitHub Discussions](https://github.com/Harky911/ReolinkANPR/discussions)

---

**Note:** This system is designed for personal use. Ensure compliance with local laws regarding video surveillance and data privacy.
