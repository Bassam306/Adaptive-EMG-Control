/*
  Master ESP32 — Final System
  ============================
  - Streams filtered EMG from CheezSEMG to PC over USB serial (230400 baud)
  - Receives gesture + effort commands from PC over same serial port
  - Forwards commands to Slave ESP32 over ESP-NOW

  Serial format (bidirectional, same port):
      ESP32 → PC  : filtered EMG float, one per line  e.g. "2.14\n"
      PC → ESP32  : "G<idx>,E<effort_pct>\n"           e.g. "G0,E78\n"

  Gesture indices:  0=Closed  1=Hook  2=Pencil  3=Rest

  Wiring:
      EMG signal  → GPIO 34 (ADC1, WiFi-safe)
      EMG detect  → GPIO 4
      USB         → PC running emg_live_final.py

  Libraries:
      CheezsEMG (already installed)
      ESP32 boards package (includes esp_now.h, WiFi.h)
*/

#include "CheezsEMG.h"
#include <esp_now.h>
#include <WiFi.h>

// ── EMG config ───────────────────────────────────────────────
#define SAMPLE_RATE  300
#define BAUD_RATE    230400
#define INPUT_PIN_1  34
#define DETECT_PIN_1 4

CheezsEMG sEMG1(INPUT_PIN_1, DETECT_PIN_1, SAMPLE_RATE);

// ── ESP-NOW config ───────────────────────────────────────────
uint8_t SLAVE_MAC[] = {0xF4, 0x65, 0x0B, 0xE7, 0x74, 0xC4};

// Packet sent to slave — must match slave struct exactly
typedef struct {
    uint8_t gesture;    // 0=Closed 1=Hook 2=Pencil 3=Rest
    uint8_t effort;     // 0–100 (percentage of MVC)
} CommandPacket;

CommandPacket packet;
bool peer_added = false;

// ── Incoming serial buffer ────────────────────────────────────
String serial_buf = "";

// ── ESP-NOW send callback (ESP32 core 3.x) ───────────────────
void onSent(const wifi_tx_info_t *info, esp_now_send_status_t status) {
    // Uncomment for debugging:
    // Serial.println(status == ESP_NOW_SEND_SUCCESS ? "OK" : "FAIL");
}

// ── Parse command from Python ─────────────────────────────────
// Format: "G<0-3>,E<0-100>\n"
bool parseCommand(String cmd, uint8_t &gesture, uint8_t &effort) {
    cmd.trim();
    if (cmd.length() < 5)     return false;
    if (cmd.charAt(0) != 'G') return false;

    int comma = cmd.indexOf(',');
    if (comma < 0)            return false;

    int g = cmd.substring(1, comma).toInt();
    int e = cmd.substring(comma + 2).toInt();   // skip 'E'

    if (g < 0 || g > 3)       return false;
    e = constrain(e, 0, 100);

    gesture = (uint8_t)g;
    effort  = (uint8_t)e;
    return true;
}

// ── Setup ─────────────────────────────────────────────────────
void setup() {
    Serial.begin(BAUD_RATE);

    // EMG
    analogReadResolution(10);
    analogSetAttenuation(ADC_11db);
    sEMG1.begin();

    // ESP-NOW
    WiFi.disconnect(true);
    WiFi.mode(WIFI_STA);
    delay(100);

    if (esp_now_init() != ESP_OK) return;
    esp_now_register_send_cb(onSent);

    esp_now_peer_info_t peer_info = {};
    memcpy(peer_info.peer_addr, SLAVE_MAC, 6);
    peer_info.channel = 0;
    peer_info.encrypt = false;

    if (esp_now_add_peer(&peer_info) == ESP_OK) {
        peer_added = true;
    }
}

// ── Main loop ─────────────────────────────────────────────────
void loop() {
    // ── Job 1: Stream EMG to Python ───────────────────────────
    if (sEMG1.checkSampleInterval()) {
        sEMG1.processSignal();
        Serial.println(sEMG1.getFilteredSignal());
    }

    // ── Job 2: Read command from Python (non-blocking) ────────
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\n') {
            if (serial_buf.length() > 0 && peer_added) {
                uint8_t gesture, effort;
                if (parseCommand(serial_buf, gesture, effort)) {
                    packet.gesture = gesture;
                    packet.effort  = effort;
                    esp_now_send(SLAVE_MAC,
                                 (uint8_t *)&packet,
                                 sizeof(packet));
                }
            }
            serial_buf = "";
        } else {
            serial_buf += c;
        }
    }
}
