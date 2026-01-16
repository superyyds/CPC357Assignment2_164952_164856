import pymongo
import paho.mqtt.client as mqtt
from datetime import datetime, timezone
import json

# ==========================================
# 1. MongoDB Configuration
# ==========================================
# Connect to local MongoDB server
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["urban_noise_db"]
collection = db["iot"]

# ==========================================
# 2. MQTT Configuration
# ==========================================
# Use "localhost" if Mosquitto is on this same VM
mqtt_broker_address = "localhost" 
mqtt_topic = "urban/noise"

# ==========================================
# 3. Callback Functions
# ==========================================

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Connected to MQTT Broker! Subscribed to: {mqtt_topic}")
        client.subscribe(mqtt_topic)
    else:
        print(f"Connection failed with code {rc}")

def on_message(client, userdata, message):
    try:
        # Decode incoming payload
        payload_str = message.payload.decode("utf-8")
        print(f"Received: {payload_str}")
        
        # Parse JSON
        data = json.loads(payload_str)
        
        # Add server-side timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        
        # Prepare document
        document = {
            "timestamp": timestamp,
            "device_id": data.get("device_id", "MakerFeatherS3"),
            "noise_type": data.get("label", "Unknown"),
            "confidence": data.get("value", 0.0),
            "raw_data": payload_str
        }
        
        # Insert into MongoDB
        collection.insert_one(document)
        print(">>> Data saved to MongoDB successfully.")
        
    except Exception as e:
        print(f"Error: {e}")

# ==========================================
# 4. Execution
# ==========================================

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

print("Starting Cloud Bridge...")
client.connect(mqtt_broker_address, 1883, 60)

try:
    client.loop_forever()
except KeyboardInterrupt:
    print("\nStopping bridge...")
    client.disconnect()