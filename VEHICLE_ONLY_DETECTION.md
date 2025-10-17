# Vehicle-Only Detection Feature

## Overview
The system now tracks **all vehicles**, even when no license plate is detected! This is useful for:
- Monitoring traffic flow
- Tracking vehicles that park in a way that obscures their plates
- Getting vehicle information even when plate recognition fails
- Building comprehensive vehicle statistics

## What's Changed

### Database Schema
- `plate_number` column now allows NULL values
- `confidence` column now allows NULL values
- Vehicle-only detections are stored with `plate_number = 'NO_PLATE'`

### Detection Logic

#### With Plate Detection (Previous Behavior)
1. Camera detects vehicle
2. System captures frames
3. FastALPR detects license plate
4. Vehicle recognition runs on the same image
5. Saves: plate number + vehicle attributes

#### Without Plate Detection (NEW)
1. Camera detects vehicle
2. System captures frames
3. FastALPR tries but finds no plate
4. **NEW**: Vehicle recognition still runs
5. Saves: 'NO_PLATE' + vehicle attributes (color, make, model)

### Deduplication Strategy

**For vehicles WITH plates:**
- Deduplication based on plate number
- Same as before: 30-second cooldown per plate

**For vehicles WITHOUT plates (NEW):**
- Deduplication based on color + make combination
- 30-second cooldown for same color/make combo
- Example: A white Toyota detected at 10:00:00 won't be saved again until 10:00:30

### Configuration

Two new settings in the ALPR configuration:

#### 1. Enable Vehicle Recognition
- **Config key**: `alpr.vehicle_recognition_enabled`
- **Default**: `True`
- **Description**: Enables color/make/model detection for ALL vehicles

#### 2. Track Vehicles Without Plates (NEW)
- **Config key**: `alpr.vehicle_only_detection_enabled`
- **Default**: `True`
- **Description**: Save vehicle info even when no plate is detected

### UI Changes

#### Dashboard Display
- Plate column now shows a **yellow badge** "No Plate" for vehicle-only detections
- Vehicle Info column shows color, make, and model (same as before)
- Images are still saved and displayed

#### Configuration Page
New toggle in ALPR settings:
```
☑ Track Vehicles Without Plates
  Save vehicle info even when no license plate is detected.
  Useful for tracking all vehicles passing by.
```

## File Naming Convention

### With Plate
```
20251016_143022_ABC123.jpg
```
Format: `{timestamp}_{plate_number}.jpg`

### Without Plate (NEW)
```
20251016_143022_NO_PLATE_red_toyota.jpg
```
Format: `{timestamp}_NO_PLATE_{color}_{make}.jpg`

## Database Examples

### Traditional Detection (With Plate)
```sql
INSERT INTO events (
    plate_number, confidence, vehicle_color, vehicle_make, vehicle_model
) VALUES (
    'ABC123', 0.95, 'red', 'toyota', 'sedan'
);
```

### Vehicle-Only Detection (NEW)
```sql
INSERT INTO events (
    plate_number, confidence, vehicle_color, vehicle_make, vehicle_model
) VALUES (
    'NO_PLATE', NULL, 'blue', 'honda', 'suv'
);
```

## Use Cases

### 1. Traffic Monitoring
Track all vehicles passing by your camera, regardless of plate visibility:
- Total vehicle count
- Peak traffic hours
- Vehicle color distribution
- Popular vehicle makes

### 2. Parking Enforcement
Monitor vehicles that park in ways that obscure their plates:
- Front-in parking (if camera sees rear)
- Covered/missing plates
- Dirty/unreadable plates

### 3. Security
Get vehicle descriptions even when plates can't be read:
- "A white SUV entered at 2 AM"
- "Black sedan circled the property 3 times"

### 4. Statistics & Analytics
Build comprehensive reports:
- Most common vehicle colors in your area
- Vehicle make/model trends
- Traffic patterns by vehicle type

## Configuration Examples

### Maximum Detection (Everything)
```yaml
alpr:
  vehicle_recognition_enabled: true
  vehicle_only_detection_enabled: true
```
**Result**: Saves vehicles with plates AND vehicles without plates

### Plate-Only Mode (Traditional)
```yaml
alpr:
  vehicle_recognition_enabled: false
  vehicle_only_detection_enabled: false
```
**Result**: Only saves vehicles when plate is detected (no vehicle attributes)

### Hybrid Mode
```yaml
alpr:
  vehicle_recognition_enabled: true
  vehicle_only_detection_enabled: false
```
**Result**: Adds vehicle attributes to plate detections, but doesn't save vehicles without plates

### Vehicle-Only Mode (No Plates)
```yaml
alpr:
  vehicle_recognition_enabled: true
  vehicle_only_detection_enabled: true
```
**Result**: Tracks all vehicles, regardless of plate detection

## Performance Considerations

### Database Size
- Vehicle-only detections take the same space as regular detections
- Expect more database entries if enabled
- No plate crop images for vehicle-only detections (saves disk space)

### Processing Time
- No additional overhead if plate is found (runs anyway)
- Minimal overhead if no plate found (single image analysis instead of none)
- Vehicle recognition: ~100-500ms per detection

### Storage Impact
Example: 100 vehicles per day
- With plates only: ~100 entries, ~200 images (full + crop)
- With vehicle-only: ~150 entries (50% more), ~150 full images + ~100 crops
- Disk space: ~20-30MB more per day

## Filtering in Dashboard

You can search/filter vehicles:

**By plate number:**
- Search: "ABC123" → finds exact matches
- Search: "NO_PLATE" → finds all vehicle-only detections

**By vehicle attributes:**
- Currently not searchable in UI (future feature)
- Can query database directly: 
  ```sql
  SELECT * FROM events WHERE vehicle_color = 'red' AND plate_number = 'NO_PLATE';
  ```

## Migration Notes

### Existing Database
- Existing entries are not affected
- Old entries will have `plate_number` and `confidence` values
- New schema is backward compatible

### Upgrading
1. Install update
2. Restart application
3. Database schema auto-updates
4. Enable feature in config if desired
5. No manual migration needed

## Troubleshooting

### "All detections show 'No Plate'"
**Possible causes:**
- Plate recognition confidence threshold too high
- Poor image quality/lighting
- Camera angle not capturing plates well

**Solution:**
- Check ALPR min_confidence setting (try lowering to 0.8)
- Adjust camera angle/position
- Review saved images to verify plates are visible

### "Vehicle attributes always 'unknown'"
**Possible causes:**
- PyTorch not installed
- Vehicle recognition disabled
- Poor image quality

**Solution:**
- Check logs for "Vehicle recognition enabled" message
- Install: `pip install torch torchvision`
- Enable in config: `vehicle_recognition_enabled: true`

### "Too many duplicate entries"
**Possible causes:**
- 30-second deduplication may be too short for slow-moving traffic
- Multiple vehicles with same color/make passing quickly

**Solution:**
- Deduplication time is hardcoded (may need code change)
- Consider filtering by time in your analytics

## Future Enhancements

Potential improvements:
- [ ] Configurable deduplication time
- [ ] Search/filter by vehicle attributes in dashboard
- [ ] Vehicle statistics dashboard
- [ ] Export vehicle reports
- [ ] Vehicle type classification (sedan, SUV, truck, etc.)
- [ ] Motion-based deduplication (track same vehicle across frames)

## API Access

Query vehicle-only detections programmatically:

```python
import aiosqlite

async def get_vehicle_only_detections():
    async with aiosqlite.connect('data/anpr.db') as db:
        cursor = await db.execute('''
            SELECT timestamp, vehicle_color, vehicle_make, vehicle_model, image_path
            FROM events
            WHERE plate_number = 'NO_PLATE'
            ORDER BY timestamp DESC
            LIMIT 100
        ''')
        return await cursor.fetchall()
```

## Summary

The vehicle-only detection feature provides comprehensive vehicle tracking beyond just license plates. It's **enabled by default** but can be toggled on/off in the configuration. This feature is particularly useful for traffic monitoring, security, and building vehicle statistics.

**Key Benefits:**
- Track ALL vehicles, not just those with readable plates
- Get vehicle descriptions for every detection
- Build better traffic statistics
- No additional setup required (works out of the box)
- Configurable (can be disabled if not needed)

---
**Last Updated**: October 16, 2025
**Feature Version**: 2.0
