/*
  Slave ESP32 — Final System
  ===========================
  Receives gesture + effort commands from Master ESP32 over ESP-NOW.
  Drives 5 MG996R servos via PCA9685 (I2C) with MVC-proportional
  adaptive feedforward control.

  Command packet from master:
      gesture : 0=Closed  1=Hook  2=Pencil  3=Rest
      effort  : 0–100 (% of MVC from Python)

  Servo orientation:
      0°   = fully CLOSED (finger flexed)
      180° = fully OPEN   (finger extended)

  PCA9685 channel mapping:
      Channel 0 = Index
      Channel 1 = Middle
      Channel 2 = Ring
      Channel 3 = Thumb
      Channel 4 = Pinky

  Adaptive feedforward control:
      Target angle  : fixed per gesture
      Step size     : BASE_STEP[gesture] × (effort / 100)
        → light contraction = slow/partial closure
        → hard contraction  = fast/full closure
      Smooth motion : incremental stepping each loop tick
                      prevents mechanical shock

  Libraries:
      Adafruit PWM Servo Driver
      (Arduino IDE → Tools → Manage Libraries → "Adafruit PWM Servo Driver")
*/

#include <esp_now.h>
#include <WiFi.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// ── PCA9685 ──────────────────────────────────────────────────
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

#define SDA_PIN      21
#define SCL_PIN      22
#define SERVOMIN     150      // calibrated 0° pulse tick
#define SERVOMAX     520      // calibrated 180° pulse tick
#define PWM_FREQ     50       // Hz
#define OSC_FREQ     27000000 // PCA9685 oscillator calibration

// ── Configuration ─────────────────────────────────────────────
#define NUM_SERVOS 5

// PCA9685 channels: [Index, Middle, Ring, Thumb, Pinky]
const int CHANNELS[NUM_SERVOS] = {0, 1, 2, 3, 4};

// Target angles per gesture (0°=closed, 180°=open)
//                               Index  Middle  Ring  Thumb  Pinky
const int TARGET_ANGLES[4][NUM_SERVOS] = {
    {  0,    0,    0,    0,    0},  // 0: Closed — all fingers fully flexed
    { 60,   60,   60,  180,   60},  // 1: Hook   — fingers curl, thumb open
    { 0,  50,  180,   0,  180},  // 2: Pencil — index+thumb pinch
    {180,  180,  180,  180,  180},  // 3: Rest   — all fingers fully extended
};

// Base step size (degrees per loop tick) per gesture
// Scaled by effort: actual_step = BASE_STEP[gesture] × (effort / 100)
const float BASE_STEP[4] = {
    8.0,   // Closed — fast, forceful
    5.0,   // Hook   — medium
    3.0,   // Pencil — slow, precise
    6.0,   // Rest   — medium return speed
};

// ── State ─────────────────────────────────────────────────────
float   current_angles[NUM_SERVOS];
uint8_t current_gesture = 3;   // start at Rest
uint8_t current_effort  = 0;   // 0–100

// ── Data structure — must match master exactly ────────────────
typedef struct {
    uint8_t gesture;
    uint8_t effort;
} CommandPacket;

// ── ESP-NOW receive callback (ESP32 core 3.x) ─────────────────
void onReceive(const esp_now_recv_info_t *info,
               const uint8_t *data, int len) {
    if (len == sizeof(CommandPacket)) {
        CommandPacket *pkt = (CommandPacket *)data;
        current_gesture = pkt->gesture;
        current_effort  = pkt->effort;
    }
}

// ── Helpers ───────────────────────────────────────────────────
uint16_t angleToPWM(float angle) {
    int a = (int)constrain(angle, 0, 180);
    return (uint16_t)map(a, 0, 180, SERVOMIN, SERVOMAX);
}

void writeServo(int finger_idx, float angle) {
    pwm.setPWM(CHANNELS[finger_idx], 0, angleToPWM(angle));
}

// ── Step servos toward current target ─────────────────────────
void stepServos() {
    // Compute step size scaled by effort
    float effort_frac = current_effort / 100.0f;
    float step = BASE_STEP[current_gesture] * effort_frac;
    step = max(step, 0.5f);   // minimum step so servos always arrive

    for (int i = 0; i < NUM_SERVOS; i++) {
        float target = (float)TARGET_ANGLES[current_gesture][i];
        float diff   = target - current_angles[i];

        if (abs(diff) < 0.5f) {
            current_angles[i] = target;
            continue;
        }

        float move        = constrain(diff, -step, step);
        current_angles[i] = constrain(current_angles[i] + move, 0.0f, 180.0f);
        writeServo(i, current_angles[i]);
    }
}

// ── Setup ─────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);

    // PCA9685 — explicit I2C pins + oscillator calibration
    Wire.begin(SDA_PIN, SCL_PIN);
    pwm.begin();
    pwm.setOscillatorFrequency(OSC_FREQ);
    pwm.setPWMFreq(PWM_FREQ);
    delay(10);

    // Start all fingers at neutral 90° — staggered to prevent power drop
    for (int i = 0; i < NUM_SERVOS; i++) {
        current_angles[i] = 90.0f;
        writeServo(i, current_angles[i]);
        delay(150);
    }

    // ESP-NOW
    WiFi.disconnect(true);
    WiFi.mode(WIFI_STA);
    delay(100);

    if (esp_now_init() != ESP_OK) {
        Serial.println("[ERROR] ESP-NOW init failed");
        return;
    }
    esp_now_register_recv_cb(onReceive);
    Serial.println("[OK] Slave ready.");
}

// ── Main loop ─────────────────────────────────────────────────
void loop() {
    stepServos();
    delay(20);   // 50 Hz servo update rate
}
