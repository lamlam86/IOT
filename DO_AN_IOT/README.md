# ĐỒ ÁN IOT - HỆ THỐNG CẢNH BÁO Ô NHIỄM KHÔNG KHÍ VÀ ÂM THANH

## Mô tả
Hệ thống IoT sử dụng AI để dự đoán và cảnh báo mức độ ô nhiễm không khí và âm thanh theo thời gian thực. Hệ thống tích hợp cảm biến, xử lý dữ liệu bằng AI (CatBoost), và hiển thị trên web dashboard.

## Thiết bị sử dụng
- **Arduino Uno**: Đọc dữ liệu từ cảm biến
- **ESP8266 12E**: Kết nối WiFi, gửi dữ liệu lên Blynk và gọi AI server
- **MQ-135**: Cảm biến chất lượng không khí (Air Quality)
- **MQ-7**: Cảm biến CO (Carbon Monoxide)
- **GP2Y1010AU0F**: Cảm biến bụi mịn PM2.5
- **KY-037**: Cảm biến âm thanh

## Cấu trúc thư mục

```
DO_AN_IOT/
├── hardware/              # Code phần cứng
│   ├── curcuit.ino       # Code Arduino (đọc cảm biến)
│   ├── esp8266.ino       # Code ESP8266 (WiFi, Blynk, AI)
│   ├── mq.h              # Header file cảm biến (nếu cần)
│   ├── schematic.png     # Sơ đồ mạch
│   └── README_ESP8266.md # Hướng dẫn ESP8266
│
├── ai_model/             # AI Model và Training
│   ├── generate_dataset_improved.py  # Tạo dataset (3000-5000 mẫu/file)
│   ├── train_pollution_model.py     # Training model CatBoost
│   ├── pollution_model.pkl          # Model đã train (99.89% accuracy)
│   ├── model_features.pkl           # Danh sách features
│   ├── charts/                     # 8 biểu đồ phân tích model
│   │   ├── 1_confusion_matrix.png
│   │   ├── 2_feature_importance.png
│   │   ├── 3_class_distribution.png
│   │   ├── 4_accuracy_per_class.png
│   │   ├── 5_prediction_probability.png
│   │   ├── 6_training_history.png
│   │   ├── 7_sensor_by_alert_level.png
│   │   └── 8_summary_dashboard.png
│   └── datasets/                   # Dataset (18,669 mẫu)
│       ├── dataset_very_clean.csv
│       ├── dataset_safe.csv
│       ├── dataset_air_polluted.csv
│       ├── dataset_noise_polluted.csv
│       ├── dataset_both_polluted.csv
│       ├── dataset_combined_all.csv
│       └── README.md
│
├── web_server/           # Web Dashboard
│   ├── web_dashboard.py  # Flask server với AI prediction
│   ├── pipeline_web.py   # Pipeline lấy dữ liệu từ Blynk
│   ├── templates/
│   │   └── dashboard.html # Giao diện web real-time
│   └── pollution_model.pkl (copy từ ai_model)
│
├── requirements.txt      # Thư viện Python cần thiết
└── README.md            # File này
```

## Hướng dẫn cài đặt

### 1. Cài đặt thư viện Python
```bash
pip install -r requirements.txt
```

### 2. Cài đặt thư viện Arduino
Trong Arduino IDE:
- **MQUnifiedsensor**: Vào Library Manager → Tìm "MQUnifiedsensor" → Install
- **SoftwareSerial**: Đã có sẵn trong Arduino core

### 3. Cấu hình Blynk
1. Tạo tài khoản tại [blynk.cloud](https://blynk.cloud)
2. Tạo project mới
3. Thêm 4 Virtual Pins: V0, V1, V2, V3
4. Copy Auth Token
5. Cập nhật token trong `hardware/esp8266.ino`:
   ```cpp
   #define BLYNK_AUTH_TOKEN "your_token_here"
   ```

### 4. Cấu hình WiFi
Sửa trong `hardware/esp8266.ino`:
```cpp
char ssid[] = "your_wifi_name";
char pass[] = "your_wifi_password";
```

### 5. Cấu hình AI Server IP
Sửa trong `hardware/esp8266.ino`:
```cpp
const char* SERVER_URL = "http://YOUR_PC_IP:5000/api/predict";
```
Tìm IP của máy tính: `ipconfig` (Windows) hoặc `ifconfig` (Linux/Mac)

## Hướng dẫn sử dụng

### Bước 1: Upload code lên Arduino
1. Mở `hardware/curcuit.ino` trong Arduino IDE
2. Chọn board: **Arduino Uno**
3. Chọn port COM
4. Upload code

### Bước 2: Upload code lên ESP8266
1. Mở `hardware/esp8266.ino` trong Arduino IDE
2. Chọn board: **NodeMCU 1.0 (ESP-12E Module)**
3. Chọn port COM
4. Upload code

### Bước 3: Training AI Model (nếu chưa có model)
```bash
cd ai_model
python generate_dataset_improved.py  # Tạo dataset
python train_pollution_model.py      # Training model
```

### Bước 4: Copy model vào web_server
```bash
# Windows
copy ai_model\pollution_model.pkl web_server\
copy ai_model\model_features.pkl web_server\

# Linux/Mac
cp ai_model/pollution_model.pkl web_server/
cp ai_model/model_features.pkl web_server/
```

### Bước 5: Chạy Web Dashboard
```bash
cd web_server
python web_dashboard.py
```

### Bước 6: Chạy Pipeline (lấy dữ liệu từ Blynk)
Mở terminal mới:
```bash
cd web_server
python pipeline_web.py
```

### Bước 7: Mở trình duyệt
Truy cập: **http://localhost:5000**

## AI Model

### Input (4 cảm biến):
- **MQ135**: Chất lượng không khí (ppm)
- **MQ7**: CO - Carbon Monoxide (ppm)
- **PM2.5**: Bụi mịn (µg/m³)
- **Sound**: Mức độ âm thanh (dB)

### Features (20 features):
- 4 giá trị cảm biến hiện tại
- 4 features thời gian (hour_sin, hour_cos, day_sin, day_cos)
- 8 lag features (giá trị trước đó)
- 4 rolling mean features (trung bình động)

### Output (4 mức cảnh báo):
- **0: TRONG LÀNH** - Không khí rất tốt + Âm thanh yên tĩnh
- **1: AN TOÀN** - Không khí tốt + Âm thanh bình thường
- **2: CẢNH BÁO** - Một trong hai yếu tố xấu
- **3: NGUY HIỂM** - Cả không khí và âm thanh đều xấu

### Độ chính xác: **99.89%**

### Dataset:
- **Tổng số mẫu**: ~18,669 mẫu
- **5 trường hợp**: Trong lành, An toàn, Ô nhiễm không khí, Ô nhiễm âm thanh, Ô nhiễm cả hai
- **Phân bố**: Mỗi trường hợp có 3000-5000 mẫu (ngẫu nhiên)

## Luồng hoạt động

1. **Arduino** đọc cảm biến → Gửi dữ liệu qua Serial sang ESP8266
2. **ESP8266** nhận dữ liệu → Gửi lên Blynk Cloud
3. **ESP8266** gọi AI server → Nhận alert level từ AI
4. **ESP8266** gửi alert level về Arduino (nếu cần điều khiển LED)
5. **Pipeline** lấy dữ liệu từ Blynk → Gửi đến Web Dashboard
6. **Web Dashboard** hiển thị real-time + AI prediction

## Sơ đồ kết nối

```
Arduino UNO          ESP8266 12E
─────────────────────────────────
Pin 12 (TX)    →     GPIO3 (RX)
Pin 13 (RX)    ←     GPIO1 (TX)
GND            ───   GND
5V             ───   VCC (hoặc 3.3V)

Cảm biến → Arduino:
- MQ135: A0
- MQ7: A1
- PM2.5: A2 (Vo), D7 (LED)
- Sound: A3
```

## Tính năng

✅ Đọc dữ liệu real-time từ 4 cảm biến  
✅ Gửi dữ liệu lên Blynk Cloud  
✅ AI dự đoán với độ chính xác 99.89%  
✅ Web dashboard hiển thị real-time  
✅ 4 mức cảnh báo chi tiết  
✅ 8 biểu đồ phân tích model  
✅ Dataset đầy đủ 5 trường hợp ô nhiễm  

## Thư viện Python cần thiết

Xem `requirements.txt` hoặc:
```bash
pip install flask pandas numpy catboost scikit-learn matplotlib seaborn joblib blynk-python
```

## Tác giả
[Điền tên của bạn]

## License
MIT

## Changelog

### Version 2.0 (Hiện tại)
- ✅ Model mới với 4 alert levels (99.89% accuracy)
- ✅ Dataset cải tiến với 18,669 mẫu
- ✅ 8 biểu đồ visualization
- ✅ Web dashboard hỗ trợ 4 levels
- ✅ Loại bỏ dependency ArduinoJson

### Version 1.0
- Model với 3 alert levels
- Dataset 2000 mẫu
- Web dashboard cơ bản
