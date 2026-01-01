# DO AN IOT - HE THONG CANH BAO O NHIEM KHONG KHI VA AM THANH

## Mo ta
He thong IoT su dung AI de du doan va canh bao muc do o nhiem khong khi va am thanh.

## Thiet bi su dung
- Arduino Uno
- ESP8266 (NodeMCU)
- MQ-135: Cam bien chat luong khong khi
- MQ-7: Cam bien CO (Carbon Monoxide)
- GP2Y1010AU0F: Cam bien bui min PM2.5
- KY-037: Cam bien am thanh

## Cau truc thu muc

```
DO_AN_IOT/
├── hardware/           # Code Arduino va ESP8266
│   ├── mq.h           # Thu vien cam bien
│   ├── curcuit.ino    # Code Arduino
│   ├── esp8266.ino    # Code ESP8266
│   ├── schematic.png  # So do mach
│   └── layout.png     # Layout PCB
│
├── ai_model/          # Model AI
│   ├── generate_dataset.py      # Tao dataset
│   ├── train_pollution_model.py # Train model
│   ├── pollution_dataset_2000.csv  # Dataset
│   ├── pollution_model.pkl      # Model da train
│   └── model_features.pkl       # Features
│
├── web_server/        # Web Dashboard
│   ├── web_dashboard.py   # Flask server
│   ├── pipeline_web.py    # Lay du lieu tu Blynk
│   ├── pollution_model.pkl
│   ├── model_features.pkl
│   └── templates/
│       └── dashboard.html # Giao dien web
│
├── requirements.txt   # Thu vien Python
└── README.md
```

## Huong dan cai dat

### 1. Cai dat thu vien Python
```bash
pip install -r requirements.txt
```

### 2. Cai dat thu vien Arduino
- MQUnifiedsensor (Library Manager)

### 3. Cau hinh Blynk
- Tao project tren Blynk
- Lay Auth Token
- Cap nhat token trong `esp8266.ino` va `pipeline_web.py`

## Huong dan su dung

### 1. Upload code len Arduino
- Mo `hardware/curcuit.ino` trong Arduino IDE
- Upload len Arduino Uno

### 2. Upload code len ESP8266
- Mo `hardware/esp8266.ino` trong Arduino IDE
- Chon board NodeMCU
- Upload len ESP8266

### 3. Chay Web Dashboard
```bash
cd web_server
python web_dashboard.py
```

### 4. Chay Pipeline
```bash
cd web_server
python pipeline_web.py
```

### 5. Mo trinh duyet
http://localhost:5000

## AI Model

### Input (4 cam bien):
- MQ135: Chat luong khong khi (ppm)
- MQ7: CO (ppm)
- PM2.5: Bui min (ug/m3)
- Sound: Am thanh (dB)

### Output (3 muc):
- 0: SAFE (An toan)
- 1: WARNING (Canh bao)
- 2: DANGER (Nguy hiem)

### Do chinh xac: 100%

## Tac gia
[Ten cua ban]

## License
MIT

