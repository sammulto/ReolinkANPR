# Enhanced Notifications with Vehicle Recognition

## Overview
Notifications now include vehicle recognition information (color, make, model) in addition to license plate data. This works for both Telegram and Home Assistant webhooks.

## Notification Formats

### Telegram Messages

#### With License Plate
```
🚗 Plate Detected: ABC123
📊 Confidence: 95.2%
Red | Toyota | Sedan
```

**Images:**
- **Vehicle Photo** - Full vehicle image with the above text as caption
- **Plate Crop** - Cropped license plate image (sent as media group/album)

#### Without License Plate (Vehicle-Only)
```
Vehicle Detected (No Plate)
Blue | Honda | SUV
```

**Images:**
- **Vehicle Photo** - Full vehicle image with the above text as caption
- **No plate crop** (since no plate was detected)

### Home Assistant Webhook

The webhook payload now includes vehicle information:

```json
{
  "plate_number": "ABC123",
  "confidence": 0.952,
  "image_path": "/path/to/image.jpg",
  "vehicle_color": "red",
  "vehicle_make": "toyota",
  "vehicle_model": "sedan"
}
```

For vehicle-only detections:
```json
{
  "plate_number": "NO_PLATE",
  "confidence": 0.0,
  "image_path": "/path/to/image.jpg",
  "vehicle_color": "blue",
  "vehicle_make": "honda",
  "vehicle_model": "suv"
}
```

## Setup

### Telegram Bot

1. **Create a bot** with @BotFather on Telegram
2. **Get your chat ID** by messaging @userinfobot
3. **Configure** in ReolinkANPR:
   - Go to http://localhost:5001/config
   - Navigate to "Notifications" tab
   - Enable Telegram
   - Enter bot token and chat ID
   - Click "Test Telegram" to verify

**Example Config:**
```yaml
notifications:
  enabled: true
  telegram:
    enabled: true
    bot_token: "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
    chat_id: "987654321"
```

### Home Assistant Webhook

1. **Create a webhook automation** in Home Assistant:
   ```yaml
   automation:
     - alias: "Vehicle Detection"
       trigger:
         - platform: webhook
           webhook_id: "reolinkranpr_detection"
       action:
         - service: notify.mobile_app_phone
           data:
             title: "Vehicle Detected"
             message: >
               Plate: {{ trigger.json.plate_number }}
               Vehicle: {{ trigger.json.vehicle_color }} {{ trigger.json.vehicle_make }} {{ trigger.json.vehicle_model }}
             data:
               image: "{{ trigger.json.image_path }}"
   ```

2. **Get the webhook URL** from Home Assistant:
   - Settings → Automations → Your automation → ⋮ → Webhook
   - Copy the URL (e.g., `https://homeassistant.local/api/webhook/reolinkranpr_detection`)

3. **Configure** in ReolinkANPR:
   - Go to http://localhost:5001/config
   - Navigate to "Notifications" tab
   - Enable Home Assistant
   - Paste webhook URL
   - Click "Test Home Assistant" to verify

**Example Config:**
```yaml
notifications:
  enabled: true
  home_assistant:
    enabled: true
    webhook_url: "https://homeassistant.local/api/webhook/reolinkranpr_detection"
```

## Home Assistant Integration Examples

### Basic Notification
```yaml
automation:
  - alias: "ANPR Detection"
    trigger:
      - platform: webhook
        webhook_id: "anpr_vehicle"
    action:
      - service: notify.mobile_app
        data:
          title: "Vehicle Detected"
          message: >
            {% if trigger.json.plate_number == "NO_PLATE" %}
              Vehicle without plate: {{ trigger.json.vehicle_color }} {{ trigger.json.vehicle_make }}
            {% else %}
              Plate: {{ trigger.json.plate_number }} ({{ (trigger.json.confidence * 100) | round(1) }}%)
              Vehicle: {{ trigger.json.vehicle_color }} {{ trigger.json.vehicle_make }} {{ trigger.json.vehicle_model }}
            {% endif %}
```

### Open Garage for Known Plates
```yaml
automation:
  - alias: "Open Garage for Known Vehicles"
    trigger:
      - platform: webhook
        webhook_id: "anpr_vehicle"
    condition:
      - condition: template
        value_template: >
          {{ trigger.json.plate_number in ['ABC123', 'XYZ789'] }}
    action:
      - service: cover.open_cover
        entity_id: cover.garage_door
      - service: notify.mobile_app
        data:
          title: "Garage Opened"
          message: "Welcome home! ({{ trigger.json.vehicle_color }} {{ trigger.json.vehicle_make }})"
```

### Turn on Lights for Specific Vehicle Colors
```yaml
automation:
  - alias: "Red Vehicle Alert"
    trigger:
      - platform: webhook
        webhook_id: "anpr_vehicle"
    condition:
      - condition: template
        value_template: "{{ trigger.json.vehicle_color == 'red' }}"
    action:
      - service: light.turn_on
        entity_id: light.driveway
        data:
          rgb_color: [255, 0, 0]  # Red
      - delay:
          seconds: 30
      - service: light.turn_off
        entity_id: light.driveway
```

### Log Vehicle Statistics
```yaml
automation:
  - alias: "Log Vehicle Statistics"
    trigger:
      - platform: webhook
        webhook_id: "anpr_vehicle"
    action:
      - service: logbook.log
        data:
          name: "Vehicle Detection"
          message: >
            Plate: {{ trigger.json.plate_number }}
            Vehicle: {{ trigger.json.vehicle_color }} {{ trigger.json.vehicle_make }} {{ trigger.json.vehicle_model }}
          entity_id: sensor.vehicle_counter
```

### Create Sensor for Last Detected Vehicle
```yaml
# configuration.yaml
template:
  - trigger:
      - platform: webhook
        webhook_id: "anpr_vehicle"
    sensor:
      - name: "Last Detected Vehicle"
        state: "{{ trigger.json.plate_number }}"
        attributes:
          color: "{{ trigger.json.vehicle_color }}"
          make: "{{ trigger.json.vehicle_make }}"
          model: "{{ trigger.json.vehicle_model }}"
          confidence: "{{ trigger.json.confidence }}"
          detected_at: "{{ now() }}"
```

### Alert for Unknown Vehicles (No Plate)
```yaml
automation:
  - alias: "Unknown Vehicle Alert"
    trigger:
      - platform: webhook
        webhook_id: "anpr_vehicle"
    condition:
      - condition: template
        value_template: "{{ trigger.json.plate_number == 'NO_PLATE' }}"
    action:
      - service: notify.mobile_app
        data:
          title: "Unknown Vehicle"
          message: >
            A {{ trigger.json.vehicle_color }} {{ trigger.json.vehicle_make }} {{ trigger.json.vehicle_model }}
            was detected but no plate could be read.
          data:
            tag: "unknown-vehicle"
            priority: high
```

## Telegram Formatting

The Telegram messages use emojis for better readability:
- Vehicle/Plate indicator
- Confidence percentage
- Vehicle color
- Vehicle make
- Vehicle model

### Media Group (Album) Feature

When a license plate is detected, Telegram receives **two images as an album**:
1. **Full vehicle image** with detection details as caption
2. **Cropped plate image** for easy reading

This makes it easier to:
- Read the plate number clearly
- See the full vehicle context
- Archive both views for records

**Note:** Vehicle-only detections (no plate) will only send the full vehicle image.

Example notification:
![Telegram Notification Example](screenshots/telegram_notification.png)

## Testing Notifications

Use the web interface to test notifications:
1. Go to http://localhost:5001/config
2. Navigate to "Notifications" tab
3. Click "Test Telegram" or "Test Home Assistant"
4. Check your phone/Home Assistant for the test message

Test message format:
```
Test notification from ReolinkANPR
```

## Notification Flow

```
Vehicle Detected
       ↓
Plate Recognition + Vehicle Recognition
       ↓
Data Saved to Database
       ↓
Notification Sent (if not duplicate)
       ↓
[Telegram] Photo with caption
[Home Assistant] Webhook with JSON payload
```

## Filtering Notifications

Currently, all detections trigger notifications. To filter:

### Option 1: Filter in Home Assistant
Use conditions in your HA automation:
```yaml
condition:
  - condition: template
    value_template: "{{ trigger.json.plate_number not in ['KNOWN1', 'KNOWN2'] }}"
```

### Option 2: Future Enhancement
We could add notification filters to the config:
```yaml
notifications:
  filters:
    only_unknown_plates: true
    only_new_vehicles: true
    min_confidence: 0.9
```

## Troubleshooting

### Telegram Not Receiving Messages
1. Check bot token is correct
2. Verify chat ID (must start a conversation with the bot first)
3. Check logs for "Sent to Telegram" message
4. Try the test button in config

### Home Assistant Not Receiving Webhooks
1. Verify webhook URL is accessible from ReolinkANPR
2. Check Home Assistant logs for webhook trigger
3. Ensure HTTPS certificates are valid (or use HTTP for local)
4. Try the test button in config

### Missing Vehicle Information
1. Ensure vehicle recognition is enabled in config
2. Check PyTorch is installed: `pip list | grep torch`
3. Look for "Vehicle recognition enabled" in logs
4. Verify vehicle attributes are being saved in database

### Duplicate Notifications
- 30-second cooldown prevents duplicates
- If vehicle leaves and returns, it will notify again after 30s
- Check logs for "Duplicate plate detected" messages

## Advanced: Custom Notification Templates

You can modify the notification format in `src/notifier.py`:

```python
# For custom Telegram format
def _format_vehicle_description(self, color, make, model):
    # Your custom formatting here
    return f"Color: {color}, Make: {make}, Model: {model}"
```

## Privacy Considerations

- **Webhook Security**: Use HTTPS for Home Assistant webhooks
- **Image Storage**: Images are stored locally, not sent to cloud services
- **Telegram**: Messages are encrypted by Telegram
- **Data Retention**: Configure database cleanup if needed

## Future Enhancements

Planned features:
- [ ] Notification templates (customizable format)
- [ ] Rich notifications with buttons (Telegram inline keyboard)
- [ ] Notification filtering (only unknown plates, etc.)
- [ ] Push notifications via other services (Discord, Slack, etc.)
- [ ] Grouped notifications (summary every X minutes)
- [ ] Vehicle recognition confidence threshold for notifications

## Support

For issues with notifications:
1. Check application logs: `logs/anpr.log`
2. Test with the built-in test buttons
3. Verify network connectivity
4. Check firewall rules for outbound HTTPS

---
**Last Updated**: October 16, 2025
**Feature**: Vehicle Recognition Enhanced Notifications
