"""Simple notification service for Home Assistant and Telegram."""

import aiohttp
from typing import Optional
from pathlib import Path
from src.logger import logger


class Notifier:
    """Handle notifications to Home Assistant and Telegram."""

    def __init__(self, config):
        self.config = config
        self.enabled = False
        self.ha_enabled = False
        self.telegram_enabled = False
        
        # Load settings
        if hasattr(config, '_config') and 'notifications' in config._config:
            notif_config = config._config['notifications']
            self.enabled = notif_config.get('enabled', False)
            
            # Home Assistant
            ha_config = notif_config.get('home_assistant', {})
            self.ha_enabled = ha_config.get('enabled', False)
            self.ha_webhook = ha_config.get('webhook_url', '')
            
            # Telegram
            tg_config = notif_config.get('telegram', {})
            self.telegram_enabled = tg_config.get('enabled', False)
            self.telegram_token = tg_config.get('bot_token', '')
            self.telegram_chat_id = tg_config.get('chat_id', '')
            
            logger.info(f"Notifications: enabled={self.enabled}, telegram={self.telegram_enabled}, ha={self.ha_enabled}")
            if self.enabled:
                logger.info("[OK] Notifications enabled")
                if self.ha_enabled:
                    logger.info("  [OK] Home Assistant webhook configured")
                if self.telegram_enabled:
                    logger.info(f"  [OK] Telegram bot configured (chat_id: {self.telegram_chat_id})")
        else:
            logger.info("Notifications: not configured")

    async def send_detection(
        self, 
        plate_number: str, 
        confidence: float, 
        image_path: Optional[str] = None,
        plate_crop_path: Optional[str] = None,
        vehicle_crop_path: Optional[str] = None,
        vehicle_color: Optional[str] = None,
        vehicle_type: Optional[str] = None,
        vehicle_confidence: Optional[float] = None
    ):
        """Send notification when a plate is detected.
        
        Args:
            plate_number: License plate number or 'NO_PLATE'
            confidence: Detection confidence
            image_path: Path to full frame image (fallback if no vehicle crop)
            plate_crop_path: Path to cropped plate image
            vehicle_crop_path: Path to cropped vehicle image (preferred for Telegram)
            vehicle_color: Detected vehicle color
            vehicle_type: Detected vehicle type (sedan, suv, etc)
            vehicle_confidence: Vehicle classification confidence (0-1)
        """
        logger.debug(f"send_detection called: enabled={self.enabled}, plate={plate_number}")
        
        if not self.enabled:
            logger.debug("Notifications disabled - skipping")
            return
        
        # Build message with vehicle info
        if plate_number == 'NO_PLATE' or 'NO_PLATE' in plate_number:
            # Vehicle-only detection
            vehicle_desc = self._format_vehicle_description(vehicle_color, vehicle_type, vehicle_confidence)
            message = f"Vehicle Detected (No Plate)\n{vehicle_desc}"
        else:
            # Normal plate detection with vehicle info
            vehicle_desc = self._format_vehicle_description(vehicle_color, vehicle_type, vehicle_confidence)
            message = f"Plate Detected: {plate_number}\nPlate Conf: {confidence:.1%}"
            if vehicle_desc:
                message += f"\n{vehicle_desc}"
        
        # Send to Home Assistant
        if self.ha_enabled and self.ha_webhook:
            logger.debug("Sending to Home Assistant...")
            await self._send_to_home_assistant(
                plate_number, confidence, image_path,
                vehicle_color, vehicle_type, vehicle_confidence
            )
        
        # Send to Telegram
        if self.telegram_enabled and self.telegram_token and self.telegram_chat_id:
            logger.debug("Sending to Telegram...")
            # Send both vehicle crop and full frame to Telegram
            await self._send_to_telegram(message, vehicle_crop_path, image_path, plate_crop_path)
        
        if not self.ha_enabled and not self.telegram_enabled:
            logger.warning("Notifications enabled but no services configured!")

    def _format_vehicle_description(
        self, 
        color: Optional[str], 
        vehicle_type: Optional[str], 
        confidence: Optional[float] = None
    ) -> str:
        """Format vehicle description for notifications."""
        parts = []
        
        if color and color != 'unknown':
            parts.append(f"{color.capitalize()}")
        
        if vehicle_type and vehicle_type != 'unknown':
            parts.append(f"{vehicle_type.capitalize()}")
        
        description = " | ".join(parts) if parts else ""
        
        return description

    async def _send_to_home_assistant(
        self, 
        plate_number: str, 
        confidence: float, 
        image_path: Optional[str],
        vehicle_color: Optional[str] = None,
        vehicle_type: Optional[str] = None,
        vehicle_confidence: Optional[float] = None
    ):
        """Send webhook to Home Assistant."""
        try:
            async with aiohttp.ClientSession() as session:
                data = {
                    'plate_number': plate_number,
                    'plate_confidence': confidence,
                    'image_path': image_path,
                    'vehicle_color': vehicle_color or 'unknown',
                    'vehicle_type': vehicle_type or 'unknown',
                    'vehicle_confidence': vehicle_confidence if vehicle_confidence is not None else 0.0
                }
                async with session.post(self.ha_webhook, json=data, timeout=5) as response:
                    if response.status == 200:
                        vehicle_info = f"{vehicle_color} {vehicle_type}".strip()
                        logger.info(f"Sent to Home Assistant: {plate_number} ({vehicle_info})")
                    else:
                        logger.warning(f"Home Assistant returned status {response.status}")
        except Exception as e:
            logger.error(f"Failed to send to Home Assistant: {e}")

    async def _send_to_telegram(self, message: str, vehicle_crop_path: Optional[str], full_image_path: Optional[str], plate_crop_path: Optional[str] = None):
        """Send messages to Telegram: full frame, vehicle crop, and optionally plate crop."""
        try:
            async with aiohttp.ClientSession() as session:
                # Check which images exist (handle None values properly)
                has_plate_crop = bool(plate_crop_path and Path(plate_crop_path).exists())
                has_vehicle_crop = bool(vehicle_crop_path and Path(vehicle_crop_path).exists())
                has_full_image = bool(full_image_path and Path(full_image_path).exists())
                
                photo_url = f"https://api.telegram.org/bot{self.telegram_token}/sendPhoto"
                
                # 1. Send full frame with vehicle info
                if has_full_image:
                    with open(full_image_path, 'rb') as img_file:
                        form = aiohttp.FormData()
                        form.add_field('chat_id', self.telegram_chat_id)
                        form.add_field('caption', f"Full Frame\n{message}", content_type='text/plain')
                        form.add_field('photo', img_file, filename='full_frame.jpg', content_type='image/jpeg')
                        
                        async with session.post(photo_url, data=form, timeout=30) as response:
                            if response.status == 200:
                                logger.info(f"Sent full frame to Telegram")
                            else:
                                response_text = await response.text()
                                logger.warning(f"Telegram full frame returned status {response.status}: {response_text}")
                
                # 2. Send vehicle crop with vehicle info
                if has_vehicle_crop:
                    with open(vehicle_crop_path, 'rb') as img_file:
                        form = aiohttp.FormData()
                        form.add_field('chat_id', self.telegram_chat_id)
                        form.add_field('caption', f"Vehicle Crop\n{message}", content_type='text/plain')
                        form.add_field('photo', img_file, filename='vehicle_crop.jpg', content_type='image/jpeg')
                        
                        async with session.post(photo_url, data=form, timeout=30) as response:
                            if response.status == 200:
                                logger.info(f"Sent vehicle crop to Telegram")
                            else:
                                response_text = await response.text()
                                logger.warning(f"Telegram vehicle crop returned status {response.status}: {response_text}")
                
                # 3. Send plate crop with vehicle info (only if plate was detected)
                if has_plate_crop:
                    with open(plate_crop_path, 'rb') as img_file:
                        form = aiohttp.FormData()
                        form.add_field('chat_id', self.telegram_chat_id)
                        form.add_field('caption', f"Plate Crop\n{message}", content_type='text/plain')
                        form.add_field('photo', img_file, filename='plate_crop.jpg', content_type='image/jpeg')
                        
                        async with session.post(photo_url, data=form, timeout=30) as response:
                            if response.status == 200:
                                logger.info(f"Sent plate crop to Telegram")
                            else:
                                response_text = await response.text()
                                logger.warning(f"Telegram plate crop returned status {response.status}: {response_text}")
                
                # If no images available, send text only
                if not has_full_image and not has_vehicle_crop and not has_plate_crop:
                    message_url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                    data = {
                        'chat_id': self.telegram_chat_id,
                        'text': message,
                        'parse_mode': 'HTML'
                    }
                    async with session.post(message_url, json=data, timeout=10) as response:
                        if response.status == 200:
                            logger.info(f"Sent text-only to Telegram: {message}")
                        else:
                            response_text = await response.text()
                            logger.warning(f"Telegram returned status {response.status}: {response_text}")
        except Exception as e:
            logger.error(f"Failed to send to Telegram: {e}")

    async def send_test(self, service: str = 'all'):
        """Send a test notification."""
        test_message = "[TEST] Test notification from ReolinkANPR"
        
        if service in ['telegram', 'all'] and self.telegram_enabled:
            logger.info("Sending test to Telegram...")
            await self._send_to_telegram(test_message, None, None)
            return True
        
        if service in ['home_assistant', 'all'] and self.ha_enabled:
            logger.info("Sending test to Home Assistant...")
            await self._send_to_home_assistant("TEST", 1.0, None)
            return True
        
        return False

