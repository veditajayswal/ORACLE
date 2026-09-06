import logging

logger = logging.getLogger("ORACLE_MQTT")

class MqttTelemetryListener:
    def __init__(self, broker_url: str = None):
        self.broker_url = broker_url
        self.is_connected = False

    def start(self):
        logger.info("MQTT listener initialized (optional protocol layer).")

    def stop(self):
        self.is_connected = False

mqtt_listener = MqttTelemetryListener()
