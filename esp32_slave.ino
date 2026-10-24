// ESP32 slave firmware - Arduino-ESP32 core 3.x
// Rx from Pi: "V,<lin m/s>,<ang rad/s>\n"  | "S\n" = stop
// Tx to Pi  : "T,<encL>,<encR>,<yaw deg>,<cliffL>,<cliffR>\n" at 20 Hz
// (core 2.x: replace ledcAttach(pin,f,res) with ledcSetup(ch,f,res)+ledcAttachPin(pin,ch)
//  and ledcWrite(pin,..) with ledcWrite(ch,..))

#include <Wire.h>

#define PWMA 25
#define AIN1 26
#define AIN2 27
#define PWMB 33
#define BIN1 32
#define BIN2 13
#define STBY 23
#define ENC_L_A 34
#define ENC_L_B 35
#define ENC_R_A 36
#define ENC_R_B 39
#define TRIG1 16
#define ECHO1 17
#define TRIG2 18
#define ECHO2 19
#define MPU_ADDR 0x68

// ---- calibrate these ----
const float TICKS_PER_REV = 660.0;   // count ticks for one wheel turn by hand
const float WHEEL_RADIUS  = 0.0425;  // m
const float WHEEL_BASE    = 0.20;    // m, wheel centre to centre
const float MAX_SPEED     = 0.60;    // m/s wheel speed at full PWM
const int   ENC_L_SIGN = 1, ENC_R_SIGN = 1;   // flip so forward = positive ticks
const int   MOT_L_SIGN = 1, MOT_R_SIGN = 1;   // flip if a wheel drives backwards
const float KP = 150.0, KI = 300.0;
const float CLIFF_CM = 12.0;         // floor farther than this = drop
const uint32_t CMD_TIMEOUT_MS = 300;

volatile long cntL = 0, cntR = 0;
void IRAM_ATTR isrL() { cntL += (digitalRead(ENC_L_A) == digitalRead(ENC_L_B)) ? 1 : -1; }
void IRAM_ATTR isrR() { cntR += (digitalRead(ENC_R_A) == digitalRead(ENC_R_B)) ? 1 : -1; }

float cmdLin = 0, cmdAng = 0;
uint32_t lastCmd = 0, lastCtl = 0, lastCliff = 0, lastTx = 0;
long prevL = 0, prevR = 0;
float intL = 0, intR = 0;
float yaw = 0, gyroBias = 0;
bool cliffL = false, cliffR = false;
uint8_t missL = 0, missR = 0, nextSensor = 0;
char buf[64]; uint8_t blen = 0;

void setMotor(int pwmPin, int in1, int in2, float u) {
  int p = constrain((int)fabs(u), 0, 255);
  if (p == 0) { digitalWrite(in1, HIGH); digitalWrite(in2, HIGH); }  // short brake
  else { digitalWrite(in1, u > 0); digitalWrite(in2, u < 0); }
  ledcWrite(pwmPin, p);
}

int16_t readGyroZ() {
  Wire.beginTransmission(MPU_ADDR); Wire.write(0x47); Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 2);
  int hi = Wire.read(); int lo = Wire.read();
  return (int16_t)((hi << 8) | lo);
}

float readCm(int trig, int echo) {
  digitalWrite(trig, LOW); delayMicroseconds(2);
  digitalWrite(trig, HIGH); delayMicroseconds(10); digitalWrite(trig, LOW);
  unsigned long d = pulseIn(echo, HIGH, 6000);
  if (d == 0) return 999.0;           // no echo = treat as drop
  return d * 0.0343f / 2.0f;
}

void cliffStep() {
  if (nextSensor == 0) {
    missL = (readCm(TRIG1, ECHO1) > CLIFF_CM) ? missL + 1 : 0;
    cliffL = missL >= 2;
  } else {
    missR = (readCm(TRIG2, ECHO2) > CLIFF_CM) ? missR + 1 : 0;
    cliffR = missR >= 2;
  }
  nextSensor ^= 1;
}

void controlStep(float dt, uint32_t now) {
  yaw += ((readGyroZ() - gyroBias) / 131.0f) * dt;

  float lin = cmdLin, ang = cmdAng;
  if (now - lastCmd > CMD_TIMEOUT_MS) { lin = 0; ang = 0; }       // watchdog
  if ((cliffL || cliffR) && lin > 0) lin = 0;                     // cliff stop

  noInterrupts(); long l = cntL, r = cntR; interrupts();
  float k = 2.0f * PI * WHEEL_RADIUS / TICKS_PER_REV / dt;
  float vL = (l - prevL) * ENC_L_SIGN * k;
  float vR = (r - prevR) * ENC_R_SIGN * k;
  prevL = l; prevR = r;

  float tL = lin - ang * WHEEL_BASE / 2.0f;
  float tR = lin + ang * WHEEL_BASE / 2.0f;

  float uL = 0, uR = 0;
  if (fabs(tL) < 0.005f) intL = 0;
  else {
    float e = tL - vL; intL = constrain(intL + e * dt, -0.5f, 0.5f);
    uL = tL / MAX_SPEED * 255.0f + KP * e + KI * intL;
  }
  if (fabs(tR) < 0.005f) intR = 0;
  else {
    float e = tR - vR; intR = constrain(intR + e * dt, -0.5f, 0.5f);
    uR = tR / MAX_SPEED * 255.0f + KP * e + KI * intR;
  }
  setMotor(PWMA, AIN1, AIN2, uL * MOT_L_SIGN);
  setMotor(PWMB, BIN1, BIN2, uR * MOT_R_SIGN);
}

void readSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      buf[blen] = 0; blen = 0;
      float a, b;
      if (sscanf(buf, "V,%f,%f", &a, &b) == 2) {
        cmdLin = constrain(a, -0.5f, 0.5f); cmdAng = constrain(b, -2.0f, 2.0f);
        lastCmd = millis();
      } else if (buf[0] == 'S') { cmdLin = 0; cmdAng = 0; lastCmd = millis(); }
    } else if (blen < sizeof(buf) - 1) buf[blen++] = c;
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(AIN1, OUTPUT); pinMode(AIN2, OUTPUT);
  pinMode(BIN1, OUTPUT); pinMode(BIN2, OUTPUT);
  pinMode(STBY, OUTPUT); digitalWrite(STBY, HIGH);
  ledcAttach(PWMA, 20000, 8); ledcAttach(PWMB, 20000, 8);
  setMotor(PWMA, AIN1, AIN2, 0); setMotor(PWMB, BIN1, BIN2, 0);
  pinMode(ENC_L_A, INPUT); pinMode(ENC_L_B, INPUT);
  pinMode(ENC_R_A, INPUT); pinMode(ENC_R_B, INPUT);
  attachInterrupt(ENC_L_A, isrL, CHANGE);
  attachInterrupt(ENC_R_A, isrR, CHANGE);
  pinMode(TRIG1, OUTPUT); pinMode(TRIG2, OUTPUT);
  pinMode(ECHO1, INPUT); pinMode(ECHO2, INPUT);

  Wire.begin(21, 22);
  Wire.beginTransmission(MPU_ADDR); Wire.write(0x6B); Wire.write(0); Wire.endTransmission();
  delay(100);
  long sum = 0;                                   // keep robot still at boot
  for (int i = 0; i < 200; i++) { sum += readGyroZ(); delay(5); }
  gyroBias = sum / 200.0f;
  lastCtl = lastCliff = lastTx = millis();
}

void loop() {
  readSerial();
  uint32_t now = millis();
  if (now - lastCtl >= 20) { controlStep((now - lastCtl) / 1000.0f, now); lastCtl = now; }
  if (now - lastCliff >= 50) { cliffStep(); lastCliff = now; }
  if (now - lastTx >= 50) {
    noInterrupts(); long l = cntL, r = cntR; interrupts();
    Serial.printf("T,%ld,%ld,%.2f,%d,%d\n", l * ENC_L_SIGN, r * ENC_R_SIGN, yaw, cliffL, cliffR);
    lastTx = now;
  }
}
