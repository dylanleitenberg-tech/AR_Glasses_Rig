// =====================================================================
//  ir_strobe.ino — IR strobe + SAFETY INTERLOCK firmware (Seeed XIAO / Arduino)
// ---------------------------------------------------------------------
//  Reference firmware for the 6-camera binocular AR eye-tracking rig. It drives the
//  940 nm IR LED rings through 2N7002 low-side MOSFETs, strobed in sync with the camera
//  shutter, and ENFORCES the eye-safety envelope in hardware+firmware so the IR is OFF
//  unless every condition is met. See WIRING.md and EYE_TRACKING.md §3.
//
//  The retina cannot feel IR, so this is built FAIL-SAFE: default OFF, and ANY anomaly
//  (host/USB drop, rail out of range, over-long pulse, blink, reset) forces the gates low.
//
//  SAFETY INTERLOCKS implemented here:
//    1. Strobe in sync with the shutter   -> IR fires only as a bounded PULSE on a host
//                                             "P" command issued at each exposure.
//    2. USB / host drop instantly kills IR -> heartbeat watchdog: no host line within
//                                             HB_TIMEOUT_MS forces both gates low.
//    3. Voltage safeguard                  -> the 5 V rail is read every loop; outside
//                                             [VMIN,VMAX] disables IR.
//    4. IR cutoff on blinks                -> host "B L|R|0" flags closed eyes; that eye's
//                                             ring is held off.
//    5. Dose / duty caps                   -> pulse width clamped to MAX_PULSE_US and a
//                                             minimum OFF time enforces a low duty cycle.
//    6. Any power anomaly -> IR off + a gate PULL-DOWN (external 100k) guarantees OFF
//                            through power-up and MCU reset.
//
//  HARDWARE NOTES (see WIRING.md):
//    * 2N7002 low-side switch per ring; gate from IR_GATE_* with an external 100k pull-down.
//    * IR branch fed from the 5 V rail THROUGH a 300 mA PTC polyfuse (overcurrent).
//    * RAIL_SENSE = the 5 V rail through a divider into an ADC pin (scale set below).
//    * IR power pair is twisted + star-grounded, separate from the USB data cables.
//    * The XIAO also bridges the IMU (I2C -> USB); that stub is at the bottom.
// =====================================================================

// ---- pins (adjust to your board) ----
const int IR_GATE_L  = D1;   // gate of the LEFT ring's 2N7002  (external 100k pull-down to GND)
const int IR_GATE_R  = D2;   // gate of the RIGHT ring's 2N7002 (external 100k pull-down to GND)
const int RAIL_SENSE = A0;   // 5 V rail via a divider (e.g. 2:1) into the ADC

// ---- safety envelope ----
const unsigned long HB_TIMEOUT_MS = 100;    // no host heartbeat within this -> IR off
const unsigned int  MAX_PULSE_US  = 2000;   // hard cap on a single IR pulse (dose cap)
const unsigned long MIN_OFF_US    = 6000;   // min gap between pulses (duty-cycle cap)
const float VMIN = 4.5, VMAX = 5.5;         // acceptable 5 V rail window (volts)

// ---- ADC scaling: reads 0..ADC_MAX over 0..ADC_VREF, rail divided by RAIL_DIV ----
const float ADC_VREF = 3.3;     // XIAO ADC reference (volts)
const int   ADC_MAX  = 1023;    // 10-bit; set 4095 if you switch to 12-bit
const float RAIL_DIV = 2.0;     // divider ratio on RAIL_SENSE (rail = reading * RAIL_DIV)

// ---- state ----
unsigned long lastHeartbeat = 0;
unsigned long lastPulseEnd  = 0;
bool enabled   = false;         // host must "EN" to arm strobing
bool blinkL    = false;
bool blinkR    = false;
String line;

void gatesOff() {
  digitalWrite(IR_GATE_L, LOW);
  digitalWrite(IR_GATE_R, LOW);
}

float railVolts() {
  int raw = analogRead(RAIL_SENSE);
  return (raw * ADC_VREF / ADC_MAX) * RAIL_DIV;
}

// every interlock that must hold for IR to be allowed AT ALL this instant
bool envelopeOK() {
  if (!enabled) return false;
  if (millis() - lastHeartbeat > HB_TIMEOUT_MS) return false;   // USB/host drop
  float v = railVolts();
  if (v < VMIN || v > VMAX) return false;                       // voltage safeguard
  return true;
}

// fire ONE bounded, interlocked IR pulse on the requested ring(s)
void pulse(unsigned int us, bool left, bool right) {
  if (!envelopeOK()) { gatesOff(); return; }
  if (micros() - lastPulseEnd < MIN_OFF_US) return;             // duty-cycle cap
  if (us > MAX_PULSE_US) us = MAX_PULSE_US;                     // dose cap
  bool l = left  && !blinkL;                                    // blink cutoff (per eye)
  bool r = right && !blinkR;
  if (!l && !r) return;
  if (l) digitalWrite(IR_GATE_L, HIGH);
  if (r) digitalWrite(IR_GATE_R, HIGH);
  unsigned long t0 = micros();
  while (micros() - t0 < us) {
    if (millis() - lastHeartbeat > HB_TIMEOUT_MS) break;        // abort mid-pulse on host drop
  }
  gatesOff();
  lastPulseEnd = micros();
}

void setup() {
  pinMode(IR_GATE_L, OUTPUT);
  pinMode(IR_GATE_R, OUTPUT);
  gatesOff();                 // fail-safe OFF on boot
  Serial.begin(115200);
  // Wire.begin();            // <- enable for the IMU I2C bridge stub below
}

void loop() {
  // ---- continuous safety: if the envelope breaks, gates go low immediately ----
  if (!envelopeOK()) gatesOff();

  // ---- parse host commands (newline-terminated) ----
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') { handleLine(line); line = ""; }
    else if (c != '\r' && line.length() < 32) line += c;
  }

  // forwardImu();            // <- enable for the IMU I2C->USB bridge
}

void handleLine(const String &s) {
  lastHeartbeat = millis();                 // ANY recognized host line is also a heartbeat
  if (s == "HB") {
    return;                                 // pure heartbeat
  } else if (s == "EN") {
    enabled = true;
  } else if (s == "DIS") {
    enabled = false; gatesOff();
  } else if (s.startsWith("P")) {           // "P <us>"  -> one strobe pulse, both rings
    int us = s.length() > 2 ? s.substring(2).toInt() : 1000;
    pulse((unsigned int)us, true, true);
  } else if (s.startsWith("B")) {           // "B L" | "B R" | "B 0"  -> blink state
    char e = s.length() > 2 ? s.charAt(2) : '0';
    blinkL = (e == 'L' || e == 'l');
    blinkR = (e == 'R' || e == 'r');
    if (blinkL) digitalWrite(IR_GATE_L, LOW);
    if (blinkR) digitalWrite(IR_GATE_R, LOW);
  } else {
    Serial.println("ERR");
  }
}

// =====================================================================
//  IMU I2C -> USB bridge (stub). The Mac has no native I2C, so the XIAO relays the
//  ICM-20948. Use SparkFun_ICM-20948 (or your driver) to read accel+gyro and Serial.print
//  them; software/imu.py's Kalman filter consumes accel-tilt + gyro-rate. Kept minimal so
//  this file stays focused on the SAFETY logic above.
// =====================================================================
// void forwardImu() {
//   // read accel (g) + gyro (deg/s) from the ICM-20948 over I2C, then:
//   // Serial.print("IMU "); Serial.print(ax); ... Serial.println(gz);
// }
