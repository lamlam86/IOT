# Pollution Dataset Collection

## Mô tả

Bộ dataset này chứa dữ liệu đầy đủ các trường hợp ô nhiễm không khí và âm thanh, được chia thành từng file riêng biệt để dễ sử dụng. Dataset được tạo với số lượng mẫu ngẫu nhiên từ 3000-5000 cho mỗi trường hợp, tổng cộng khoảng 18,000-25,000 mẫu.

## Các file dataset

### 1. `dataset_very_clean.csv` (~3000-5000 samples)
**Trường hợp: Trong lành**
- Không khí rất tốt + Âm thanh yên tĩnh
- MQ135: 5-25 ppm
- CO: 0.1-0.8 ppm
- PM2.5: 0-20 µg/m³
- Sound: 30-42 dB
- Alert Level: 0 (Very Clean)

### 2. `dataset_safe.csv` (~3000-5000 samples)
**Trường hợp: An toàn**
- Không khí tốt + Âm thanh bình thường
- MQ135: 20-50 ppm
- CO: 0.5-2.0 ppm
- PM2.5: 10-40 µg/m³
- Sound: 40-55 dB
- Alert Level: 1 (Safe)

### 3. `dataset_air_polluted.csv` (~3000-5000 samples)
**Trường hợp: Ô nhiễm không khí**
- Không khí xấu + Âm thanh tốt
- MQ135: 60-150 ppm
- CO: 3.0-15.0 ppm
- PM2.5: 60-180 µg/m³
- Sound: 35-50 dB
- Alert Level: 2-3 (Warning/Danger)

### 4. `dataset_noise_polluted.csv` (~3000-5000 samples)
**Trường hợp: Ô nhiễm âm thanh**
- Không khí tốt + Âm thanh ồn
- MQ135: 15-45 ppm
- CO: 0.3-1.5 ppm
- PM2.5: 5-35 µg/m³
- Sound: 65-90 dB
- Alert Level: 2-3 (Warning/Danger)

### 5. `dataset_both_polluted.csv` (~3000-5000 samples)
**Trường hợp: Ô nhiễm cả hai**
- Cả không khí và âm thanh đều xấu
- MQ135: 100-200 ppm
- CO: 8.0-30.0 ppm
- PM2.5: 120-300 µg/m³
- Sound: 75-100 dB
- Alert Level: 3 (Danger)

### 6. `dataset_combined_all.csv` (~18,000-25,000 samples)
**File tổng hợp**
- Chứa tất cả mẫu từ 5 trường hợp trên
- Đã được shuffle ngẫu nhiên
- **Sử dụng file này để training model chính**

## Cấu trúc dữ liệu

Mỗi file CSV chứa các cột sau:

- `DateTime`: Thời gian đo (datetime)
- `Hour`: Giờ trong ngày (0-23)
- `DayOfWeek`: Ngày trong tuần (0=Monday, 6=Sunday)
- `Month`: Tháng (1-12)
- `MQ135_AirQuality`: Chất lượng không khí (ppm)
- `MQ7_CO_ppm`: Nồng độ CO (ppm)
- `PM25_ugm3`: Bụi mịn PM2.5 (µg/m³)
- `Sound_dB`: Mức độ âm thanh (dB)
- `AirQuality_Label`: Nhãn chất lượng không khí (0-4)
  - 0: Very Good
  - 1: Good
  - 2: Moderate
  - 3: Unhealthy
  - 4: Very Unhealthy
- `Noise_Label`: Nhãn mức độ ồn (0-4)
  - 0: Very Quiet
  - 1: Quiet
  - 2: Normal
  - 3: Noisy
  - 4: Very Noisy
- `Alert_Level`: Mức cảnh báo (0-3)
  - 0: Very Clean (Trong lành)
  - 1: Safe (An toàn)
  - 2: Warning (Cảnh báo)
  - 3: Danger (Nguy hiểm)
- `Is_Air_Polluted`: Binary flag (1 nếu ô nhiễm không khí)
- `Is_Noise_Polluted`: Binary flag (1 nếu ô nhiễm âm thanh)
- `Scenario`: Mã trường hợp (VERY_CLEAN, SAFE, AIR_POLLUTED, NOISE_POLLUTED, BOTH_POLLUTED)
- `Scenario_Name`: Tên trường hợp

## Cách sử dụng

### Training model với dataset tổng hợp:
```python
import pandas as pd
df = pd.read_csv('datasets/dataset_combined_all.csv')
# ... training code ...
```

### Training model với từng trường hợp riêng:
```python
# Chỉ training với dữ liệu ô nhiễm không khí
df_air = pd.read_csv('datasets/dataset_air_polluted.csv')

# Chỉ training với dữ liệu ô nhiễm âm thanh
df_noise = pd.read_csv('datasets/dataset_noise_polluted.csv')
```

### Kết hợp nhiều trường hợp:
```python
df1 = pd.read_csv('datasets/dataset_very_clean.csv')
df2 = pd.read_csv('datasets/dataset_safe.csv')
df3 = pd.read_csv('datasets/dataset_air_polluted.csv')
df_combined = pd.concat([df1, df2, df3], ignore_index=True)
```

## Thống kê

- **Tổng số mẫu**: ~18,000-25,000 (mỗi lần generate khác nhau)
- **Phân bố ngẫu nhiên**: Mỗi trường hợp có 3000-5000 mẫu (không đều)
- **Phạm vi giá trị**: Dựa trên dữ liệu thực tế từ cảm biến
- **Pattern thời gian**: Có tính đến giờ cao điểm, cuối tuần, mùa
- **Realistic data**: Dữ liệu có pattern thực tế, không phải random thuần túy

## Tạo dataset mới

Để tạo lại dataset với số lượng mẫu ngẫu nhiên:

```bash
cd ai_model
python generate_dataset_improved.py
```

Script sẽ tự động:
- Tạo số mẫu ngẫu nhiên từ 3000-5000 cho mỗi trường hợp
- Phân bố không đều (tự nhiên hơn)
- Lưu vào thư mục `datasets/`

## Lưu ý

- Dataset được tạo bằng script `generate_dataset_improved.py`
- Sử dụng seed=42 để có thể tái tạo (nhưng số lượng mẫu vẫn random)
- Dữ liệu có pattern thực tế (giờ cao điểm, mùa, v.v.)
- Alert Level được tính dựa trên cả Air Quality và Noise
- File CSV có thể rất lớn (không commit lên git, chỉ giữ README)

## Model Training

Sau khi có dataset, training model:

```bash
cd ai_model
python train_pollution_model.py
```

Model sẽ được lưu tại:
- `pollution_model.pkl` - Model đã train
- `model_features.pkl` - Danh sách features
- `charts/` - 8 biểu đồ phân tích
