#ifndef MQ_H
#define MQ_H

#include <Arduino.h>
#include <MQUnifiedsensor.h>

// ==================== CẤU HÌNH ====================
#define BOARD "Arduino UNO"
#define VOLTAGE_RESOLUTION 5.0
#define ADC_BIT_RESOLUTION 10

#define RATIO_MQ135_CLEAN_AIR 3.6
#define RATIO_MQ7_CLEAN_AIR 27.5

// ==================== MQ135 SENSOR ====================
class MQ135_Sensor {
  private:
    MQUnifiedsensor* _mq;
  public:
    MQ135_Sensor(int pin) {
      _mq = new MQUnifiedsensor(BOARD, VOLTAGE_RESOLUTION, ADC_BIT_RESOLUTION, pin, "MQ-135");
    }
    
    void begin() {
      _mq->setRegressionMethod(1);
      _mq->setA(110.47);
      _mq->setB(-2.862);
      _mq->init();
    }
    
    void calibrate(int samples = 10) {
      float calcR0 = 0;
      for (int i = 0; i < samples; i++) {
        _mq->update();
        calcR0 += _mq->calibrate(RATIO_MQ135_CLEAN_AIR);
        delay(100);
      }
      _mq->setR0(calcR0 / samples);
    }
    
    float getAirQuality() {
      _mq->update();
      return _mq->readSensor();
    }
    
    float getR0() { return _mq->getR0(); }
};

// ==================== MQ7 SENSOR (CO) ====================
class MQ7_Sensor {
  private:
    MQUnifiedsensor* _mq;
  public:
    MQ7_Sensor(int pin) {
      _mq = new MQUnifiedsensor(BOARD, VOLTAGE_RESOLUTION, ADC_BIT_RESOLUTION, pin, "MQ-7");
    }
    
    void begin() {
      _mq->setRegressionMethod(1);
      _mq->setA(99.042);
      _mq->setB(-1.518);
      _mq->init();
    }
    
    void calibrate(int samples = 10) {
      float calcR0 = 0;
      for (int i = 0; i < samples; i++) {
        _mq->update();
        calcR0 += _mq->calibrate(RATIO_MQ7_CLEAN_AIR);
        delay(100);
      }
      _mq->setR0(calcR0 / samples);
    }
    
    float getCO_PPM() {
      _mq->update();
      return _mq->readSensor();
    }
    
    float getR0() { return _mq->getR0(); }
};

// ==================== DUST SENSOR GP2Y1010AU0F (TỰ VIẾT - KHÔNG CẦN THƯ VIỆN) ====================
class DustSensor {
  private:
    int _analogPin;
    int _ledPin;
  public:
    DustSensor(int analogPin, int ledPin) : _analogPin(analogPin), _ledPin(ledPin) {}
    
    void begin() {
      pinMode(_ledPin, OUTPUT);
      digitalWrite(_ledPin, HIGH);
    }
    
    float getDensity() {
      digitalWrite(_ledPin, LOW);
      delayMicroseconds(280);
      int val = analogRead(_analogPin);
      delayMicroseconds(40);
      digitalWrite(_ledPin, HIGH);
      delayMicroseconds(9680);
      
      float voltage = val * (5.0 / 1024.0);
      float dust = (voltage - 0.9) * 200.0;
      return (dust < 0) ? 0 : dust;
    }
};

// ==================== SOUND SENSOR KY-037 ====================
class SoundSensor {
  private:
    int _pin;
  public:
    SoundSensor(int pin) : _pin(pin) {}
    
    float getDecibel() {
      unsigned long start = millis();
      int sMax = 0, sMin = 1024;
      
      while (millis() - start < 50) {
        int s = analogRead(_pin);
        if (s > sMax) sMax = s;
        if (s < sMin) sMin = s;
      }
      
      return map(sMax - sMin, 0, 700, 35, 95);
    }
};

#endif
