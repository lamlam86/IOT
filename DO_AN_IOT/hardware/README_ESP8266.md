# CAI DAT THU VIEN CHO ESP8266

## Thu vien can cai:
1. **Blynk** (da co)
2. **ArduinoJson** - Cai tu Library Manager
3. **ESP8266HTTPClient** (da co trong ESP8266 core)

## Buoc cai:
1. Mo Arduino IDE
2. Sketch → Include Library → Manage Libraries
3. Tim "ArduinoJson" → Install (version 6.x)

## Cau hinh IP:
- Mo file `esp8266.ino`
- Tim dong: `const char* SERVER_URL = "http://192.168.1.55:5000/api/predict";`
- Thay IP `192.168.1.55` bang IP cua may ban neu khac

## Luu y:
- Web server phai dang chay (`python web_dashboard.py`)
- ESP8266 va may tinh phai cung mang WiFi

