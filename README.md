# ReolinkANPR

Automatic Number Plate Recognition (ANPR) system for Reolink cameras with AI vehicle detection **and vehicle recognition**.

---

## What's New in v2.0

### Vehicle Recognition (October 2025)
- **Vehicle Color Detection** - Automatically detects vehicle colors using computer vision
- **Make & Model Recognition** - Identifies vehicle manufacturer and model using deep learning
- **Vehicle-Only Tracking** - Tracks vehicles even when no plate is detected
- **Enhanced Notifications** - Telegram & Home Assistant now include full vehicle information
- **Smart Deduplication** - Separate 30s cooldown for plate vs. vehicle-based detections
- **Database Migration** - Automatically upgrades existing databases with new columns
- **Configurable** - Easy toggle in web UI, no code changes needed

**Upgrade Notes:**
- Existing databases auto-migrate on startup
- PyTorch optional but recommended: `pip install torch torchvision`
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
- **Vehicle Color** - Black, White, Silver, Red, Blue, Green, Yellow, etc. (Very accurate!)
- **Vehicle Make** - Toyota, Ford, Honda, Chevrolet, BMW, Mercedes, etc.
- **Vehicle Model** - Sedan, SUV, Truck, etc.

### Key Features
- **Track ALL Vehicles** - Even when no license plate is detected
- **Enhanced Notifications** - Telegram and Home Assistant receive full vehicle info
- **Smart Deduplication** - Separate logic for plates vs. vehicle-only detections
- **Optional & Configurable** - Toggle on/off via web interface
- **GPU Acceleration** - Automatically uses CUDA if available

### Example Dashboard Entry
```
Plate: ABC123          Vehicle: Red | Toyota | Sedan
```

### Example Notification (Telegram)
```
🚗 Plate Detected: ABC123
📊 Confidence: 95.2%
Red | Toyota | Sedan
[Vehicle Photo]
```

**See [VEHICLE_RECOGNITION.md](VEHICLE_RECOGNITION.md) for complete documentation.**

## Features

- ✅ **Real-time Vehicle Detection** - Uses Reolink's built-in AI via TCP push events (Baichuan protocol)
- ✅ **Automatic License Plate Recognition** - Powered by FastALPR with YOLO detection
- ✅ **Vehicle Recognition** - Detects vehicle color, make, and model using deep learning
- ✅ **Vehicle-Only Tracking** - Tracks all vehicles even when no plate is detected
- ✅ **Smart Deduplication** - Prevents duplicate entries with 30-second cooldown (plate-based & vehicle-based)
- ✅ **Smart Recording** - Optional camera settings optimization for plate capture
- ✅ **Web Dashboard** - Modern dark/light theme UI with plate crops, vehicle info, and full images
- ✅ **Enhanced Notifications** - Telegram & Home Assistant webhooks with vehicle information
- ✅ **In-App Documentation** - Built-in guides for all settings
- ✅ **Zero Polling** - Event-driven architecture for instant detection
- ✅ **High Performance** - Processes 20 FPS with minimal CPU usage

## Quick Start

### Prerequisites

- **Python 3.11+** (Required for TCP push event support)
- **Reolink Camera** with AI vehicle detection enabled
- **PyTorch** (Optional, for vehicle recognition - automatically uses CPU or GPU)
- **ffmpeg** (for video recording)
  ```bash
  # Ubuntu/Debian
  sudo apt install ffmpeg python3.11 python3.11-venv
  
  # macOS
  brew install ffmpeg python@3.11
  ```

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Harky911/ReolinkANPR.git
   cd ReolinkANPR
   ```

2. **Run the setup script:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```
   
   This will:
   - Create a Python virtual environment
   - Install all dependencies
   - Create necessary directories
   - Generate example configuration

3. **Start the service:**
   ```bash
   python run.py
   ```

4. **Configure your camera via web UI:**
   - Open `http://localhost:5001/config`
   - Enter your camera details (IP, username, password)
   - Save and restart the service
   - All settings configurable through the web interface!

5. **Access the web dashboard:**
   ```
   http://localhost:5001
   ```

## Configuration

### Quick Setup via Dashboard

**Step 1: Access Configuration Page**
```bash
# Start the service
python run.py

# Open in your browser
http://localhost:5001/config
```

**Step 2: Configure Camera Settings**
1. **Camera Name:** Give your camera a friendly name (e.g., "Front Door")
2. **Camera IP/Host:** Enter your camera's IP address (e.g., `192.168.1.100`)
3. **Username:** Usually `admin`
4. **Password:** Your camera password (**Important:** Change from default!)
5. **Camera Channel:** 
   - `0` for standalone cameras
   - `0, 1, 2...` for NVR channels
6. **Recording Duration:** `6` seconds (adjust based on vehicle speed)

**Step 3: Configure ALPR Settings**
1. **Detection Model:** Keep default `YOLO v9 Tiny 640`
2. **OCR Model:** Keep default `CCT-S v1 Global`
3. **Min Confidence:** `0.90` (90% confidence threshold)
4. **Vehicle Recognition:** ✅ Enabled (detects color, make, model)
5. **Track Vehicles Without Plates:** ✅ Enabled (tracks all vehicles)

**Step 4: Save and Restart**
1. Click **"Save Configuration"**
2. Stop the service (Ctrl+C)
3. Restart: `python run.py`

**That's it!** Visit `http://localhost:5001` to see detections.

### Optional: Camera Control Settings

Fine-tune camera ISP settings in real-time:

1. Scroll to **"Camera Control Settings"**
2. Adjust settings and click **"Apply Camera Settings"**:
   - **Exposure Mode:** Auto (recommended) or Manual
   - **Day/Night Mode:** Black & White (recommended for plates)
   - **Gain/Shutter:** Only visible in Manual mode

**💡 Tip:** Black & White mode works best for plate reading!

### Optional: Notifications

Enable Home Assistant or Telegram notifications with vehicle information:

1. Scroll to **"Notifications"**
2. Check **"Enable Notifications"**
3. Configure your preferred method:
   - **Home Assistant:** Enter webhook URL (receives full vehicle data)
   - **Telegram:** Enter bot token and chat ID (receives photo with vehicle info)
4. Click **"Save Configuration"**

**Enhanced Notifications Include:**
- Plate number (or "No Plate" for vehicle-only detections)
- Confidence score
- Vehicle color (e.g., "Red", "Blue", "Black")
- Vehicle make (e.g., "Toyota", "Honda", "Ford")
- Vehicle model (e.g., "Sedan", "SUV")

**Example Telegram Message:**
```
🚗 Plate Detected: ABC123
📊 Confidence: 95.2%
🎨 Red | 🏭 Toyota | 🚙 Sedan
[Vehicle Photo]
```

See [NOTIFICATIONS.md](NOTIFICATIONS.md) for detailed setup and automation examples.

### Optional: Advanced Recording Settings

Optimize camera settings automatically during recording:

1. Scroll to **"Advanced Recording Settings"**
2. **Enable Before Recording Settings:**
   - Exposure: Manual
   - Day/Night: Black & White
   - Gain: 1-40
   - Shutter: 1-4 (fast shutter for moving vehicles)
3. **Enable After Recording Settings:**
   - Restore to Auto settings
4. Click **"Save Recording Settings"**

**💡 See In-App Documentation** (`http://localhost:5001/docs`) for detailed guides!

## Web Dashboard

The web dashboard provides:

### Main Dashboard
- Real-time detection feed
- Recent plate recognitions
- **Vehicle information** (color, make, model) displayed as badges
- **"No Plate" detections** shown with yellow badge
- Confidence scores and timestamps
- Full-size vehicle images and plate crops
- Searchable plate history

### Configuration Page
- Camera settings management
- Real-time ISP control (exposure, gain, shutter, etc.)
- Recording settings configuration
- ALPR model selection
- **Vehicle Recognition** - Toggle color/make/model detection
- **Vehicle-Only Tracking** - Enable tracking without plates
- **Notifications** - Configure Telegram & Home Assistant webhooks
- All settings configurable via web UI (no manual YAML editing)

### Documentation Page
- Built-in guides for every setting
- Quick tips for optimal ANPR performance
- Home Assistant automation examples
- Telegram bot setup instructions
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
