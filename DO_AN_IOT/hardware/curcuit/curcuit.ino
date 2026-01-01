#include <MQUnifiedsensor.h>
#include <SoftwareSerial.h>

// SoftwareSerial gửi sang ESP8266
// Arduino TX(12) -> ESP8266 RX
// Arduino RX(13) <- ESP8266 TX
SoftwareSerial espSerial(13, 12);

// Cấu hình MQ
#define Board                 ("Arduino UNO")
#define Voltage_Resolution    (5)
#define ADC_Bit_Resolution    (10)

// MQ135
#define Pin135                (A0)
#define Type135               ("MQ-135")
MQUnifiedsensor MQ135(Board, Voltage_Resolution, ADC_Bit_Resolution, Pin135, Type135);

// MQ7
#define Pin7                  (A1)
#define Type7                 ("MQ-7")
MQUnifiedsensor MQ7(Board, Voltage_Resolution, ADC_Bit_Resolution, Pin7, Type7);

// Dust Sensor (tự viết)
#define DUST_PIN              A2
#define LED_PIN               7

// Sound Sensor
#define SOUND_PIN             A3

// Bien luu alert level tu AI
int aiAlertLevel = 0;

float getDustDensity() {
  digitalWrite(LED_PIN, LOW);
  delayMicroseconds(280);
  int val = analogRead(DUST_PIN);
  delayMicroseconds(40);
  digitalWrite(LED_PIN, HIGH);
  delayMicroseconds(9680);
  
  float voltage = val * (5.0 / 1024.0);
  float dust = (voltage - 0.9) * 200.0;
  return (dust < 0) ? 0 : dust;
}

float getDecibel() {
  unsigned long start = millis();
  int sMax = 0, sMin = 1024;
  while (millis() - start < 30) {
    int s = analogRead(SOUND_PIN);
    if (s > sMax) sMax = s;
    if (s < sMin) sMin = s;
  }
  return map(sMax - sMin, 0, 700, 35, 95);
}

void checkESP8266Response() {
  // Doc phan hoi tu ESP8266
  while (espSerial.available() > 0) {
    String response = espSerial.readStringUntil('\n');
    response.trim();
    
    // Kiem tra format "ALERT:X"
    if (response.startsWith("ALERT:")) {
      int colonIndex = response.indexOf(':');
      if (colonIndex > 0) {
        String levelStr = response.substring(colonIndex + 1);
        levelStr.trim();
        aiAlertLevel = levelStr.toInt();
      }
    }
  }
}

void setup() {
  Serial.begin(9600);
  espSerial.begin(9600);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);
  
  Serial.println(F("=== IoT Air & Sound Monitor ==="));
  
  // MQ135
  MQ135.setRegressionMethod(1);
  MQ135.setA(110.47);
  MQ135.setB(-2.862);
  MQ135.init();
  
  Serial.println(F("Calibrating MQ135..."));
  float calcR0_135 = 0;
  for (int i = 0; i < 10; i++) {
    MQ135.update();
    calcR0_135 += MQ135.calibrate(3.6);
    delay(100);
  }
  MQ135.setR0(calcR0_135 / 10);
  Serial.print(F("MQ135 R0 = ")); Serial.println(MQ135.getR0());
  
  // MQ7
  MQ7.setRegressionMethod(1);
  MQ7.setA(99.042);
  MQ7.setB(-1.518);
  MQ7.init();
  
  Serial.println(F("Calibrating MQ7..."));
  float calcR0_7 = 0;
  for (int i = 0; i < 10; i++) {
    MQ7.update();
    calcR0_7 += MQ7.calibrate(27.5);
    delay(100);
  }
  MQ7.setR0(calcR0_7 / 10);
  Serial.print(F("MQ7 R0 = ")); Serial.println(MQ7.getR0());
  
  Serial.println(F("San sang!"));
}

void loop() {
  // Doc phan hoi tu ESP8266 (doc nhieu lan de khong bo sot)
  checkESP8266Response();
  
  MQ135.update();
  float valueMQ135 = MQ135.readSensor();
  
  MQ7.update();
  float valueMQ7 = MQ7.readSensor();
  
  float valueDust = getDustDensity();
  float valueNoise = getDecibel();
  
  Serial.print(valueMQ135); Serial.print(",");
  Serial.print(valueMQ7);   Serial.print(",");
  Serial.print(valueDust);  Serial.print(",");
  Serial.print(valueNoise);
  Serial.print(" | AI Alert: "); Serial.println(aiAlertLevel);
  
  espSerial.print(valueMQ135); espSerial.print(",");
  espSerial.print(valueMQ7);   espSerial.print(",");
  espSerial.print(valueDust);  espSerial.print(",");
  espSerial.println(valueNoise);
  
  delay(50);
}