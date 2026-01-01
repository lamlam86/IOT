# HƯỚNG DẪN ESP8266

## Thư viện cần cài

### 1. Blynk
- Vào **Sketch → Include Library → Manage Libraries**
- Tìm "**Blynk**" → Install (version 1.0.1 hoặc mới hơn)

### 2. ESP8266 Board Support
- Vào **File → Preferences**
- Thêm URL: `http://arduino.esp8266.com/stable/package_esp8266com_index.json`
- Vào **Tools → Board → Boards Manager**
- Tìm "**ESP8266**" → Install

### 3. ESP8266HTTPClient
- Đã có sẵn trong ESP8266 core, không cần cài thêm

## Cấu hình

### 1. WiFi
Sửa trong `esp8266.ino`:
```cpp
char ssid[] = "ten_wifi_cua_ban";
char pass[] = "mat_khau_wifi";
```

### 2. Blynk Auth Token
1. Đăng nhập [blynk.cloud](https://blynk.cloud)
2. Tạo project mới
3. Thêm 4 Virtual Pins: **V0, V1, V2, V3**
4. Copy Auth Token
5. Sửa trong `esp8266.ino`:
```cpp
#define BLYNK_AUTH_TOKEN "your_token_here"
```

### 3. AI Server IP
1. Tìm IP máy tính chạy web server:
   - Windows: `ipconfig` → Tìm IPv4 Address
   - Linux/Mac: `ifconfig` → Tìm inet
2. Sửa trong `esp8266.ino`:
```cpp
const char* SERVER_URL = "http://192.168.1.55:5000/api/predict";
```
Thay `192.168.1.55` bằng IP của máy bạn

### 4. Board Settings
Trong Arduino IDE:
- **Board**: NodeMCU 1.0 (ESP-12E Module)
- **Upload Speed**: 115200
- **CPU Frequency**: 80 MHz
- **Flash Size**: 4MB (FS:2MB OTA:~1019KB)
- **Port**: Chọn COM port của ESP8266

## Kết nối phần cứng

```
ESP8266 12E          Arduino UNO
─────────────────────────────────
GPIO1 (TX)    →      Pin 13 (RX)
GPIO3 (RX)    ←      Pin 12 (TX)
GND           ───    GND
VCC           ───    5V (hoặc 3.3V nếu ESP8266 chỉ nhận 3.3V)
```

**Lưu ý**: ESP8266 chạy 3.3V, nếu Arduino cấp 5V thì cần dùng level shifter hoặc chia áp.

## Upload code

1. Kết nối ESP8266 vào máy tính qua USB
2. Chọn đúng COM port
3. Nhấn **Upload**
4. Mở Serial Monitor (9600 baud) để xem log

## Kiểm tra hoạt động

Sau khi upload, Serial Monitor sẽ hiển thị:
```
--- He thong bat dau ---
Dang ket noi WiFi...
>>> Da ket noi WiFi!
IP: 192.168.1.xxx
>>> Da ket noi Blynk thanh cong!
```

Nếu thấy lỗi:
- **WiFi không kết nối**: Kiểm tra SSID và password
- **Blynk không kết nối**: Kiểm tra Auth Token
- **AI server không gọi được**: Kiểm tra IP và đảm bảo web server đang chạy

## Luồng dữ liệu

1. ESP8266 nhận dữ liệu từ Arduino qua Serial
2. ESP8266 gửi dữ liệu lên Blynk Cloud (V0, V1, V2, V3)
3. ESP8266 gọi AI server mỗi 1 giây
4. ESP8266 nhận alert level từ AI
5. ESP8266 gửi alert level về Arduino qua Serial (format: "ALERT:2\n")

## Troubleshooting

### ESP8266 không kết nối WiFi
- Kiểm tra SSID và password
- Đảm bảo WiFi 2.4GHz (ESP8266 không hỗ trợ 5GHz)
- Kiểm tra khoảng cách đến router

### ESP8266 không gọi được AI server
- Kiểm tra IP server có đúng không
- Đảm bảo web server đang chạy (`python web_dashboard.py`)
- Kiểm tra firewall có chặn port 5000 không
- ESP8266 và máy tính phải cùng mạng WiFi

### ESP8266 không nhận dữ liệu từ Arduino
- Kiểm tra kết nối TX/RX
- Đảm bảo cả hai đều dùng baud rate 9600
- Kiểm tra Serial Monitor của Arduino có đang mở không (chỉ một Serial Monitor được mở)

### Blynk không nhận dữ liệu
- Kiểm tra Auth Token
- Kiểm tra Virtual Pins (V0, V1, V2, V3) đã được thêm vào project chưa
- Kiểm tra kết nối internet của ESP8266

## Code Structure

```cpp
void setup() {
  // Kết nối WiFi
  // Kết nối Blynk
}

void loop() {
  Blynk.run();  // Duy trì kết nối Blynk
  
  if (Serial.available()) {
    // Đọc dữ liệu từ Arduino
    // Parse dữ liệu
    // Gửi lên Blynk
    // Gọi AI server
    // Gửi alert level về Arduino
  }
}
```

## Performance

- **Tần suất gọi AI**: Mỗi 1 giây (PREDICT_INTERVAL = 1000ms)
- **HTTP Timeout**: 2 giây
- **Baud Rate**: 9600 (với Arduino)

## Lưu ý quan trọng

⚠️ **Không dùng ArduinoJson**: Code đã tự parse JSON thủ công để tránh lỗi thư viện

⚠️ **Serial Communication**: ESP8266 dùng hardware Serial để giao tiếp với Arduino, không dùng SoftwareSerial

⚠️ **Power Supply**: Đảm bảo cấp đủ nguồn cho ESP8266 (ít nhất 500mA)
