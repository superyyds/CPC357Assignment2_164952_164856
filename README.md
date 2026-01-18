# 🏙️ Urban Noise Monitoring System

An end-to-end IoT solution designed to detect and classify urban noise pollution such as sirens, drilling, car horns using **Edge AI** and **Cloud Computing**. This assignment supports **SDG 3: Good Health and Well-being: Sustainable Cities and Communities** & **SDG 11: Sustainable Cities and Communities** by providing data-driven insights to mitigate noise pollution while preserving citizen privacy.

## 🚀 Assignment Overview
This system utilizes a **Maker Feather S3 (ESP32-S3)** to capture environmental audio and classify it locally using a **TinyML** model. Detected noise metadata is transmitted via **MQTT** to a **Google Cloud Platform (GCP)** VM, where a Python-based middleware bridge stores the data in a **NoSQL MongoDB** database for historical analysis and urban planning.

### 🏗️ System Architecture
1.  **Perception Layer:** INMP441 I2S Microphone captures raw environmental audio.
2.  **Edge Layer:** ESP32-S3 runs an Edge Impulse TinyML model to classify sounds locally.
3.  **Network Layer:** Metadata is published to an MQTT Broker (Mosquitto) hosted on a GCP VM.
4.  **Middleware Layer:** Python script (`mongo_bridge_gcp.py`) handles the ingestion from MQTT to Database.
5.  **Application Layer:** MongoDB stores persistent records for longitudinal noise studies.



---

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| **`cpcproject_inference.ino`** | Main Arduino sketch for ESP32-S3. Handles audio capture, TinyML inference, and MQTT publishing. |
| **`mongo_bridge_gcp.py`** | Python middleware bridge that connects the MQTT Broker to the MongoDB instance. |
| **`ei-cpc357-project-arduino-1.0.9.zip`** | Exported Edge Impulse C++ library containing the trained urban noise classification model. |
| **`dataset_prep.py`** | Data engineering script used for cleaning and organizing the UrbanSound8K dataset. |
| **`background_noise.ipynb`** | Jupyter Notebook for feature extraction, signal processing, and model validation. |
| **`requirement.txt`** | Python dependency list (pymongo, paho-mqtt) for the cloud environment. |

---

## 🛠️ Setup and Installation

### 1. Hardware Configuration
* Import the `.zip` library into your Arduino IDE (**Sketch > Include Library > Add .ZIP Library**).
* Open `cpcproject_inference.ino`.
* Update your WiFi credentials and the `mqtt_server` variable with your **GCP External IP**.
* Connect the INMP441 microphone to the I2S pins specified in the code and upload.

### 2. Cloud Server Setup (GCP VM)
* **MQTT Broker:** Ensure Mosquitto is installed and configured to allow external connections on port 1883.
* **Database:** Ensure MongoDB is active.
* **Python Environment:**
    ```bash
    pip3 install -r requirement.txt --break-system-packages
    ```
* **Run the Bridge:**
    ```bash
    python3 mongo_bridge_gcp.py
    ```

### 3. Data Verification
To verify that noise detections are being stored correctly, enter the MongoDB shell:
```bash
mongosh
use urban_noise_db
db.iot.find().pretty()
```

### 👥 Group 30 Details 
* **University:** Universiti Sains Malaysia (USM) 
* **Course:** CPC357 - IoT Architecture & Smart Applications 
* **Group Members:**
    1. Marcus Tan Tung Chean - 164952 
    2. Ng Zi Jian - 164856 

