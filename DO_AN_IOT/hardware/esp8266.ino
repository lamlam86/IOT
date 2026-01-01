#define BLYNK_TEMPLATE_ID "TMPL6ViaKq-T2"
#define BLYNK_TEMPLATE_NAME "iot"
#define BLYNK_AUTH_TOKEN "x473czYVGJ7ogu1s8I2OFamOHW3evXkX"

#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <BlynkSimpleEsp8266.h>

char ssid[] = "iicha"; 
char pass[] = "iicha84b1"; 
char auth[] = BLYNK_AUTH_TOKEN;

// Web server URL
const char* SERVER_URL = "http://192.168.1.55:5000/api/predict";

WiFiClient client;
HTTPClient http;

unsigned long lastPredictTime = 0;
const unsigned long PREDICT_INTERVAL = 1000;  // Goi API moi 1 giay (real-time)

void setup() {
  Serial.begin(9600);
  
  WiFi.begin(ssid, pass);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }
  
  Blynk.begin(auth, ssid, pass);
}

void sendAlertToArduino(int alertLevel) {
  // Gui alert level ve Arduino qua Serial (hardware Serial cua ESP8266)
  // Format: "ALERT:2\n"
  Serial.print("ALERT:");
  Serial.println(alertLevel);
  delay(10);  // Cho Serial gui xong
}

int parseAlertLevel(String jsonResponse) {
  // Tim "alert_level": trong JSON
  int alertIndex = jsonResponse.indexOf("\"alert_level\":");
  if (alertIndex < 0) {
    return -1;
  }
  
  // Tim so sau dau :
  int startIndex = alertIndex + 14; // Do dai cua "alert_level":
  int endIndex = startIndex;
  
  // Tim dau phay hoac dau ngoac
  while (endIndex < jsonResponse.length()) {
    char c = jsonResponse.charAt(endIndex);
    if (c == ',' || c == '}' || c == ' ') {
      break;
    }
    endIndex++;
  }
  
  // Lay chuoi so
  String levelStr = jsonResponse.substring(startIndex, endIndex);
  levelStr.trim();
  
  return levelStr.toInt();
}

int getAlertFromAI(float mq135, float mq7, float pm25, float sound) {
  if (WiFi.status() != WL_CONNECTED) {
    return -1;
  }
  
  http.begin(client, SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(2000);  // Timeout 2 giay
  
  // Tao JSON request
  String jsonData = "{";
  jsonData += "\"mq135\":" + String(mq135) + ",";
  jsonData += "\"mq7\":" + String(mq7) + ",";
  jsonData += "\"pm25\":" + String(pm25) + ",";
  jsonData += "\"sound\":" + String(sound);
  jsonData += "}";
  
  int httpCode = http.POST(jsonData);
  
  if (httpCode == 200) {
    String response = http.getString();
    int alertLevel = parseAlertLevel(response);
    http.end();
    return alertLevel;
  } else {
    http.end();
    return -1;
  }
}

void loop() {
  Blynk.run();
  
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    data.trim();
    
    int firstComma  = data.indexOf(',');
    int secondComma = data.indexOf(',', firstComma + 1);
    int thirdComma  = data.indexOf(',', secondComma + 1);
    
    if (firstComma > 0 && secondComma > firstComma && thirdComma > secondComma) {
      float mq135_val = data.substring(0, firstComma).toFloat();
      float mq7_val   = data.substring(firstComma + 1, secondComma).toFloat();
      float dust_val  = data.substring(secondComma + 1, thirdComma).toFloat();
      float noise_val = data.substring(thirdComma + 1).toFloat();
      
      // Gui len Blynk
      Blynk.virtualWrite(V0, mq135_val);
      Blynk.virtualWrite(V1, mq7_val);
      Blynk.virtualWrite(V2, dust_val);
      Blynk.virtualWrite(V3, noise_val);
      
      // Goi AI server moi 1 giay
      if (millis() - lastPredictTime >= PREDICT_INTERVAL) {
        int alertLevel = getAlertFromAI(mq135_val, mq7_val, dust_val, noise_val);
        
        if (alertLevel >= 0) {
          // Gui alert level ve Arduino
          sendAlertToArduino(alertLevel);
        }
        
        lastPredictTime = millis();
      }
    }
  }
}
