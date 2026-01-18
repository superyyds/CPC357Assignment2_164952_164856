#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <driver/i2s.h>
#include "time.h"
#include <CPC357_Project_inferencing.h> 

// --- CONFIGURATION ---
const char* ssid = "YOUR SSID";
const char* password = "YOUR PASSWORD";
const char* mqtt_server = "MQTT SERVER EXTERNAL IP"; 
const char* device_id = "makerfeathers301";
const char* mqtt_topic = "urban/noise";

// I2S Microphone Settings (INMP441)
#define I2S_WS 38
#define I2S_SD 39
#define I2S_SCK 40
#define I2S_PORT I2S_NUM_0

// PIN DEFINITIONS
#define PIN_LED_GREEN A1 
#define PIN_LED_RED   A0  

// Variables for blinking without delay()
unsigned long previousMillis = 0;
bool ledState = LOW;

WiFiClient espClient;
PubSubClient client(espClient);

// Audio Buffer
int16_t *inference_buffer; 

void setup_external_leds() {
    pinMode(PIN_LED_GREEN, OUTPUT);
    pinMode(PIN_LED_RED, OUTPUT);
    
    // Quick Test Sequence
    digitalWrite(PIN_LED_GREEN, HIGH); delay(500);
    digitalWrite(PIN_LED_RED, HIGH);   delay(500);
    digitalWrite(PIN_LED_GREEN, LOW);
    digitalWrite(PIN_LED_RED, LOW);
}

void trigger_alert(String label, float confidence) {
    unsigned long currentMillis = millis();

    // 1. DANGER DETECTED (Siren or Drilling)
    if ((label == "siren" || label == "drilling") && confidence > 0.60) {
        digitalWrite(PIN_LED_GREEN, LOW);
        
        // STROBE RED (Fast 50ms blink)
        if (currentMillis - previousMillis >= 50) {
            previousMillis = currentMillis;
            ledState = !ledState; 
            digitalWrite(PIN_LED_RED, ledState);
        }
    }
    // 2. SAFE / BACKGROUND
    else {
        digitalWrite(PIN_LED_RED, LOW);

        // HEARTBEAT GREEN (Slow 1000ms blink)
        if (currentMillis - previousMillis >= 1000) {
            previousMillis = currentMillis;
            ledState = !ledState;
            digitalWrite(PIN_LED_GREEN, ledState);
        }
    }
}

void setup() {
    Serial.begin(115200);

    Serial.println("--- NOISE MONITOR INITIALIZING ---");
    setup_external_leds();
    
    // Connect WiFi
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi Connected");

    // Sync Time for MQTT Timestamps
    configTime(28800, 0, "pool.ntp.org", "time.google.com"); 
    
    // Setup MQTT
    client.setServer(mqtt_server, 1883);

    // Allocate Audio Buffer
    inference_buffer = (int16_t *)malloc(EI_CLASSIFIER_RAW_SAMPLE_COUNT * sizeof(int16_t));
    if (!inference_buffer) {
        Serial.println("ERR: Buffer Allocation Failed!");
        while(1);
    }

    // I2S Microphone Initialization
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = 16000, 
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 64,
        .use_apll = false
    };
    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = -1,
        .data_in_num = I2S_SD
    };
    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_PORT, &pin_config);
}

void capture_audio() {
    size_t bytesIn = 0;
    i2s_read(I2S_PORT, (char *)inference_buffer, EI_CLASSIFIER_RAW_SAMPLE_COUNT * sizeof(int16_t), &bytesIn, portMAX_DELAY);
}

int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
    for (size_t i = 0; i < length; i++) {
        out_ptr[i] = (float)inference_buffer[offset + i];
    }
    return 0;
}

void preprocess_audio(int16_t *buffer, size_t size) {
    long sum = 0;
    for (size_t i = 0; i < size; i++) sum += buffer[i];
    int16_t dc_offset = sum / size;
    for (size_t i = 0; i < size; i++) buffer[i] -= dc_offset;
}

bool is_loud_enough_rms(int16_t *buffer, size_t size, float threshold) {
    double sum_squares = 0;
    for (size_t i = 0; i < size; i++) sum_squares += (double)buffer[i] * buffer[i];
    float rms = sqrt(sum_squares / size);
    Serial.print("Volume RMS: "); Serial.println(rms);
    return (rms > threshold);
}

void loop() {
    // 1. MQTT Reconnect
    if (!client.connected()) {
        if (client.connect(device_id)) {
            Serial.println("MQTT Connected to Cloud");
        }
    }
    client.loop();

    // 2. Capture and Filter Silence
    Serial.println("Listening...");
    capture_audio();
    preprocess_audio(inference_buffer, EI_CLASSIFIER_RAW_SAMPLE_COUNT);

    if (!is_loud_enough_rms(inference_buffer, EI_CLASSIFIER_RAW_SAMPLE_COUNT, 3150)) {
        trigger_alert("background", 0.0);
        return; 
    }

    // 3. TinyML Inference
    signal_t signal;
    signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
    signal.get_data = &raw_feature_get_data;
    ei_impulse_result_t result;

    if (run_classifier(&signal, &result, false) == EI_IMPULSE_OK) {
        
        StaticJsonDocument<256> doc;
        doc["device_id"] = device_id;
        
        String detected_label = "";
        float highest_confidence = 0.0;

        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            if (result.classification[ix].value > highest_confidence) {
                highest_confidence = result.classification[ix].value;
                detected_label = result.classification[ix].label;
            }
        }

        // 4. LED Feedback
        trigger_alert(detected_label, highest_confidence);
        
        // 5. Publish to Cloud if it's a target noise
        if (highest_confidence > 0.70 && detected_label != "background") {
            doc["label"] = detected_label;
            doc["value"] = highest_confidence;
            
            char jsonBuffer[256];
            serializeJson(doc, jsonBuffer);
            client.publish(mqtt_topic, jsonBuffer);
            Serial.print("🚀 Data sent to MongoDB: "); Serial.println(jsonBuffer);
        }
    }
}
