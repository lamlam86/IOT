"""
Generate Dataset cho AI dự đoán ô nhiễm không khí và âm thanh
Dựa trên dữ liệu thực từ cảm biến: MQ135, MQ7, PM2.5, Sound

Output thực tế của bạn:
MQ135: 9-65 ppm (Air Quality)
MQ7: 0.59-1.31 ppm (CO)
PM2.5: 37-77 µg/m³
Sound: 35-36 dB
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Seed để tái tạo được
np.random.seed(42)
random.seed(42)

# Số lượng mẫu (2000 dòng, mỗi 30 phút 1 mẫu = khoảng 42 ngày)
N_SAMPLES = 2000

print("=" * 60)
print("GENERATING DATASET FOR AIR & SOUND POLLUTION PREDICTION")
print("=" * 60)

# Tạo datetime index - từ ngày hiện tại trở về trước
end_date = datetime(2026, 1, 1, 12, 0, 0)  # Ngày hiện tại
start_date = end_date - timedelta(minutes=30 * N_SAMPLES)
timestamps = [start_date + timedelta(minutes=30 * i) for i in range(N_SAMPLES)]

# Tạo DataFrame
df = pd.DataFrame({'DateTime': timestamps})
df['Year'] = df['DateTime'].dt.year
df['Month'] = df['DateTime'].dt.month
df['Day'] = df['DateTime'].dt.day
df['Hour'] = df['DateTime'].dt.hour
df['DayOfWeek'] = df['DateTime'].dt.dayofweek
df['WeekOfYear'] = df['DateTime'].dt.isocalendar().week

# ==================== GENERATE REALISTIC SENSOR DATA ====================

print("\nGenerating sensor data based on your real output...")

# --- MQ135 (Air Quality) - Range: 5-200 ppm ---
# Pattern: Cao vào giờ cao điểm, thấp ban đêm
base_mq135 = 30  # Baseline từ dữ liệu thực của bạn (~10-65)

mq135_values = []
for i, row in df.iterrows():
    hour = row['Hour']
    month = row['Month']
    dow = row['DayOfWeek']
    
    # Base value
    val = base_mq135
    
    # Giờ cao điểm sáng (7-9h): +20-40
    if 7 <= hour <= 9:
        val += np.random.uniform(20, 40)
    # Giờ cao điểm chiều (17-20h): +25-50
    elif 17 <= hour <= 20:
        val += np.random.uniform(25, 50)
    # Ban đêm (22h-6h): -10-20
    elif hour >= 22 or hour <= 6:
        val -= np.random.uniform(5, 15)
    else:
        val += np.random.uniform(-5, 15)
    
    # Cuối tuần ít ô nhiễm hơn
    if dow >= 5:
        val *= 0.8
    
    # Mùa đông (tháng 11-2) ô nhiễm hơn
    if month in [11, 12, 1, 2]:
        val *= 1.3
    
    # Random noise
    val += np.random.normal(0, 5)
    val = max(5, min(200, val))  # Clamp
    
    mq135_values.append(round(val, 2))

df['MQ135_AirQuality'] = mq135_values


# --- MQ7 (CO) - Range: 0.1-50 ppm ---
# Pattern: Tương quan với MQ135, nhưng range nhỏ hơn
base_co = 0.8  # Từ dữ liệu thực của bạn (~0.6-1.3)

co_values = []
for i, row in df.iterrows():
    hour = row['Hour']
    dow = row['DayOfWeek']
    
    val = base_co
    
    # Giờ cao điểm
    if 7 <= hour <= 9:
        val += np.random.uniform(0.5, 2.0)
    elif 17 <= hour <= 20:
        val += np.random.uniform(0.8, 3.0)
    elif hour >= 22 or hour <= 6:
        val -= np.random.uniform(0.1, 0.3)
    else:
        val += np.random.uniform(-0.2, 0.5)
    
    # Cuối tuần
    if dow >= 5:
        val *= 0.7
    
    # Correlation với MQ135
    val += (mq135_values[i] - base_mq135) * 0.02
    
    val += np.random.normal(0, 0.2)
    val = max(0.1, min(50, val))
    
    co_values.append(round(val, 2))

df['MQ7_CO_ppm'] = co_values


# --- PM2.5 (Dust) - Range: 0-300 µg/m³ ---
# Pattern: Từ dữ liệu thực của bạn (~37-77)
base_pm25 = 45

pm25_values = []
for i, row in df.iterrows():
    hour = row['Hour']
    month = row['Month']
    dow = row['DayOfWeek']
    
    val = base_pm25
    
    # Giờ cao điểm
    if 7 <= hour <= 9:
        val += np.random.uniform(15, 40)
    elif 17 <= hour <= 20:
        val += np.random.uniform(20, 50)
    elif hour >= 22 or hour <= 6:
        val -= np.random.uniform(5, 15)
    else:
        val += np.random.uniform(-10, 20)
    
    # Cuối tuần
    if dow >= 5:
        val *= 0.75
    
    # Mùa đông ô nhiễm hơn
    if month in [11, 12, 1, 2]:
        val *= 1.4
    # Mùa mưa (5-9) sạch hơn
    elif month in [5, 6, 7, 8, 9]:
        val *= 0.7
    
    # Random events (đốt rác, xây dựng...)
    if random.random() < 0.05:  # 5% chance
        val += np.random.uniform(50, 150)
    
    val += np.random.normal(0, 10)
    val = max(0, min(300, val))
    
    pm25_values.append(round(val, 2))

df['PM25_ugm3'] = pm25_values


# --- Sound (dB) - Range: 30-100 dB ---
# Pattern: Từ dữ liệu thực của bạn (~35)
base_sound = 40

sound_values = []
for i, row in df.iterrows():
    hour = row['Hour']
    dow = row['DayOfWeek']
    
    val = base_sound
    
    # Ban ngày ồn hơn
    if 6 <= hour <= 22:
        val += np.random.uniform(10, 25)
    else:
        val -= np.random.uniform(5, 10)
    
    # Giờ cao điểm
    if 7 <= hour <= 9 or 17 <= hour <= 19:
        val += np.random.uniform(5, 15)
    
    # Cuối tuần yên tĩnh hơn ban ngày nhưng ồn hơn ban đêm
    if dow >= 5:
        if 10 <= hour <= 22:
            val += np.random.uniform(-5, 10)
    
    # Random events (xe cứu thương, xây dựng...)
    if random.random() < 0.03:  # 3% chance
        val += np.random.uniform(20, 40)
    
    val += np.random.normal(0, 3)
    val = max(30, min(100, val))
    
    sound_values.append(round(val, 1))

df['Sound_dB'] = sound_values


# ==================== CREATE LABELS (Pollution Level) ====================

print("Creating pollution labels...")

def get_air_quality_label(mq135, co, pm25):
    """
    Đánh giá mức độ ô nhiễm không khí
    0: Tốt (Good)
    1: Trung bình (Moderate)
    2: Kém (Unhealthy for Sensitive)
    3: Xấu (Unhealthy)
    4: Rất xấu (Very Unhealthy)
    """
    # Tính AQI đơn giản dựa trên PM2.5 và CO
    aqi_pm25 = 0
    if pm25 <= 12:
        aqi_pm25 = pm25 * 50 / 12
    elif pm25 <= 35.4:
        aqi_pm25 = 50 + (pm25 - 12) * 50 / 23.4
    elif pm25 <= 55.4:
        aqi_pm25 = 100 + (pm25 - 35.4) * 50 / 20
    elif pm25 <= 150.4:
        aqi_pm25 = 150 + (pm25 - 55.4) * 50 / 95
    else:
        aqi_pm25 = 200 + (pm25 - 150.4) * 100 / 100
    
    aqi_co = 0
    if co <= 4.4:
        aqi_co = co * 50 / 4.4
    elif co <= 9.4:
        aqi_co = 50 + (co - 4.4) * 50 / 5
    elif co <= 12.4:
        aqi_co = 100 + (co - 9.4) * 50 / 3
    else:
        aqi_co = 150 + (co - 12.4) * 50 / 3
    
    aqi = max(aqi_pm25, aqi_co)
    
    if aqi <= 50:
        return 0  # Good
    elif aqi <= 100:
        return 1  # Moderate
    elif aqi <= 150:
        return 2  # Unhealthy for Sensitive
    elif aqi <= 200:
        return 3  # Unhealthy
    else:
        return 4  # Very Unhealthy

def get_noise_label(db):
    """
    Đánh giá mức độ ồn
    0: Yên tĩnh (< 45 dB)
    1: Bình thường (45-55 dB)
    2: Ồn (55-70 dB)
    3: Rất ồn (> 70 dB)
    """
    if db < 45:
        return 0
    elif db < 55:
        return 1
    elif db < 70:
        return 2
    else:
        return 3

# Apply labels
df['AirQuality_Label'] = df.apply(
    lambda row: get_air_quality_label(row['MQ135_AirQuality'], row['MQ7_CO_ppm'], row['PM25_ugm3']), 
    axis=1
)
df['Noise_Label'] = df['Sound_dB'].apply(get_noise_label)

# Combined pollution alert (0: Safe, 1: Warning, 2: Danger)
def get_alert_level(air_label, noise_label):
    if air_label >= 3 or noise_label >= 3:
        return 2  # Danger
    elif air_label >= 2 or noise_label >= 2:
        return 1  # Warning
    else:
        return 0  # Safe

df['Alert_Level'] = df.apply(
    lambda row: get_alert_level(row['AirQuality_Label'], row['Noise_Label']),
    axis=1
)


# ==================== SAVE DATASET ====================

# Reorder columns
columns_order = [
    'DateTime', 'Year', 'Month', 'Day', 'Hour', 'DayOfWeek', 'WeekOfYear',
    'MQ135_AirQuality', 'MQ7_CO_ppm', 'PM25_ugm3', 'Sound_dB',
    'AirQuality_Label', 'Noise_Label', 'Alert_Level'
]
df = df[columns_order]

# Save to CSV
output_file = 'pollution_dataset_2000.csv'
df.to_csv(output_file, index=False)

print(f"\n[OK] Dataset saved to: {output_file}")
print(f"   Total samples: {len(df)}")
print(f"   Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")

# Statistics
print("\n" + "=" * 60)
print("DATASET STATISTICS")
print("=" * 60)

print("\n📊 Sensor Value Ranges:")
print(f"   MQ135 (Air Quality): {df['MQ135_AirQuality'].min():.2f} - {df['MQ135_AirQuality'].max():.2f} ppm")
print(f"   MQ7 (CO):            {df['MQ7_CO_ppm'].min():.2f} - {df['MQ7_CO_ppm'].max():.2f} ppm")
print(f"   PM2.5:               {df['PM25_ugm3'].min():.2f} - {df['PM25_ugm3'].max():.2f} µg/m³")
print(f"   Sound:               {df['Sound_dB'].min():.1f} - {df['Sound_dB'].max():.1f} dB")

print("\n📈 Label Distribution:")
print("\nAir Quality Labels:")
print(df['AirQuality_Label'].value_counts().sort_index())
print("\nNoise Labels:")
print(df['Noise_Label'].value_counts().sort_index())
print("\nAlert Levels:")
print(df['Alert_Level'].value_counts().sort_index())

print("\n" + "=" * 60)
print("DONE! Use this dataset to train your AI model.")
print("=" * 60)

