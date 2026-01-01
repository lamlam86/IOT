"""
Generate Dataset đầy đủ các trường hợp ô nhiễm
Chia thành từng file riêng biệt cho mỗi trường hợp:
1. Trong lành (Very Clean) - Không khí rất tốt + Âm thanh yên tĩnh
2. An toàn (Safe) - Không khí tốt + Âm thanh bình thường
3. Ô nhiễm không khí (Air Polluted) - Không khí xấu + Âm thanh tốt
4. Ô nhiễm âm thanh (Noise Polluted) - Không khí tốt + Âm thanh ồn
5. Ô nhiễm cả hai (Both Polluted) - Cả hai đều xấu
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Seed để tái tạo được
np.random.seed(42)
random.seed(42)

# Mỗi file có từ 3000-5000 mẫu ngẫu nhiên
MIN_SAMPLES_PER_SCENARIO = 3000
MAX_SAMPLES_PER_SCENARIO = 5000

print("=" * 70)
print("GENERATING COMPLETE DATASET FOR POLLUTION PREDICTION")
print("=" * 70)
print(f"Each scenario will have {MIN_SAMPLES_PER_SCENARIO}-{MAX_SAMPLES_PER_SCENARIO} samples")

print("=" * 70)
print("GENERATING COMPLETE DATASET FOR POLLUTION PREDICTION")
print("=" * 70)

# Tạo thư mục output nếu chưa có
output_dir = 'datasets'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"\n[OK] Created directory: {output_dir}")

# ==================== ĐỊNH NGHĨA CÁC TRƯỜNG HỢP ====================

scenarios = {
    'VERY_CLEAN': {
        'name': 'Trong lanh',
        'description': 'Khong khi rat tot + Am thanh yen tinh',
        'mq135_range': (5, 25),
        'co_range': (0.1, 0.8),
        'pm25_range': (0, 20),
        'sound_range': (30, 42),
        'weight': 0.15
    },
    'SAFE': {
        'name': 'An toan',
        'description': 'Khong khi tot + Am thanh binh thuong',
        'mq135_range': (20, 50),
        'co_range': (0.5, 2.0),
        'pm25_range': (10, 40),
        'sound_range': (40, 55),
        'weight': 0.30
    },
    'AIR_POLLUTED': {
        'name': 'O nhiem khong khi',
        'description': 'Khong khi xau + Am thanh tot',
        'mq135_range': (60, 150),
        'co_range': (3.0, 15.0),
        'pm25_range': (60, 180),
        'sound_range': (35, 50),
        'weight': 0.25
    },
    'NOISE_POLLUTED': {
        'name': 'O nhiem am thanh',
        'description': 'Khong khi tot + Am thanh on',
        'mq135_range': (15, 45),
        'co_range': (0.3, 1.5),
        'pm25_range': (5, 35),
        'sound_range': (65, 90),
        'weight': 0.20
    },
    'BOTH_POLLUTED': {
        'name': 'O nhiem ca hai',
        'description': 'Ca khong khi va am thanh deu xau',
        'mq135_range': (100, 200),
        'co_range': (8.0, 30.0),
        'pm25_range': (120, 300),
        'sound_range': (75, 100),
        'weight': 0.10
    }
}

# ==================== HÀM TẠO DỮ LIỆU CHO MỘT TRƯỜNG HỢP ====================

def generate_scenario_data(scenario_key, scenario_config, n_samples):
    """Tao du lieu cho mot truong hop cu the"""
    
    print(f"\n[{scenario_key}] Generating {n_samples} samples: {scenario_config['name']}")
    print(f"    Description: {scenario_config['description']}")
    
    # Tạo timestamps
    end_date = datetime.now()
    start_date = end_date - timedelta(minutes=30 * n_samples)
    timestamps = [start_date + timedelta(minutes=30 * i) for i in range(n_samples)]
    
    df = pd.DataFrame({'DateTime': timestamps})
    df['Hour'] = df['DateTime'].dt.hour
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    df['Month'] = df['DateTime'].dt.month
    
    # Tạo dữ liệu cảm biến với pattern thực tế
    mq135_values = []
    co_values = []
    pm25_values = []
    sound_values = []
    
    for i, row in df.iterrows():
        hour = row['Hour']
        dow = row['DayOfWeek']
        month = row['Month']
        
        # MQ135 với pattern theo giờ
        base_mq135 = np.random.uniform(*scenario_config['mq135_range'])
        if 7 <= hour <= 9 or 17 <= hour <= 20:
            base_mq135 += np.random.uniform(0, (scenario_config['mq135_range'][1] - scenario_config['mq135_range'][0]) * 0.2)
        elif hour >= 22 or hour <= 6:
            base_mq135 -= np.random.uniform(0, (scenario_config['mq135_range'][1] - scenario_config['mq135_range'][0]) * 0.1)
        
        if dow >= 5:  # Cuối tuần
            base_mq135 *= 0.9
        
        base_mq135 = max(scenario_config['mq135_range'][0], 
                        min(scenario_config['mq135_range'][1], base_mq135))
        mq135_values.append(round(base_mq135, 2))
        
        # CO - tương quan với MQ135
        base_co = np.random.uniform(*scenario_config['co_range'])
        co_correlation = (base_mq135 - scenario_config['mq135_range'][0]) / \
                         (scenario_config['mq135_range'][1] - scenario_config['mq135_range'][0])
        base_co += co_correlation * (scenario_config['co_range'][1] - scenario_config['co_range'][0]) * 0.3
        base_co = max(scenario_config['co_range'][0], 
                     min(scenario_config['co_range'][1], base_co))
        co_values.append(round(base_co, 2))
        
        # PM2.5
        base_pm25 = np.random.uniform(*scenario_config['pm25_range'])
        if 7 <= hour <= 9 or 17 <= hour <= 20:
            base_pm25 += np.random.uniform(0, (scenario_config['pm25_range'][1] - scenario_config['pm25_range'][0]) * 0.15)
        
        if month in [11, 12, 1, 2]:  # Mùa đông
            base_pm25 *= 1.2
        elif month in [5, 6, 7, 8, 9]:  # Mùa mưa
            base_pm25 *= 0.8
        
        base_pm25 = max(scenario_config['pm25_range'][0], 
                       min(scenario_config['pm25_range'][1], base_pm25))
        pm25_values.append(round(base_pm25, 2))
        
        # Sound
        base_sound = np.random.uniform(*scenario_config['sound_range'])
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            base_sound += np.random.uniform(0, (scenario_config['sound_range'][1] - scenario_config['sound_range'][0]) * 0.2)
        elif hour >= 22 or hour <= 6:
            base_sound -= np.random.uniform(0, (scenario_config['sound_range'][1] - scenario_config['sound_range'][0]) * 0.15)
        
        if dow >= 5:  # Cuối tuần
            if 10 <= hour <= 22:
                base_sound += np.random.uniform(0, 5)
        
        base_sound = max(scenario_config['sound_range'][0], 
                        min(scenario_config['sound_range'][1], base_sound))
        sound_values.append(round(base_sound, 1))
    
    df['MQ135_AirQuality'] = mq135_values
    df['MQ7_CO_ppm'] = co_values
    df['PM25_ugm3'] = pm25_values
    df['Sound_dB'] = sound_values
    
    # Thêm label
    df['Scenario'] = scenario_key
    df['Scenario_Name'] = scenario_config['name']
    
    return df

# ==================== TẠO LABELS ====================

def get_air_quality_label(mq135, co, pm25):
    """0: Very Good, 1: Good, 2: Moderate, 3: Unhealthy, 4: Very Unhealthy"""
    score = 0
    
    # PM2.5 (quan trọng nhất)
    if pm25 <= 12:
        score += 0
    elif pm25 <= 35.4:
        score += 1
    elif pm25 <= 55.4:
        score += 2
    elif pm25 <= 150.4:
        score += 3
    else:
        score += 4
    
    # CO
    if co <= 4.4:
        score += 0
    elif co <= 9.4:
        score += 1
    elif co <= 12.4:
        score += 2
    elif co <= 15.4:
        score += 3
    else:
        score += 4
    
    # MQ135
    if mq135 <= 30:
        score += 0
    elif mq135 <= 50:
        score += 1
    elif mq135 <= 80:
        score += 2
    elif mq135 <= 120:
        score += 3
    else:
        score += 4
    
    avg_score = score / 3
    
    if avg_score < 0.3:
        return 0  # Very Good
    elif avg_score < 1.0:
        return 1  # Good
    elif avg_score < 2.0:
        return 2  # Moderate
    elif avg_score < 3.0:
        return 3  # Unhealthy
    else:
        return 4  # Very Unhealthy

def get_noise_label(db):
    """0: Very Quiet, 1: Quiet, 2: Normal, 3: Noisy, 4: Very Noisy"""
    if db < 40:
        return 0  # Very Quiet
    elif db < 50:
        return 1  # Quiet
    elif db < 60:
        return 2  # Normal
    elif db < 75:
        return 3  # Noisy
    else:
        return 4  # Very Noisy

def get_alert_level(air_label, noise_label, scenario):
    """0: Very Clean, 1: Safe, 2: Warning, 3: Danger"""
    if scenario == 'VERY_CLEAN':
        return 0  # Very Clean
    elif scenario == 'SAFE':
        return 1  # Safe
    elif scenario in ['AIR_POLLUTED', 'NOISE_POLLUTED']:
        if air_label >= 3 or noise_label >= 3:
            return 3  # Danger
        else:
            return 2  # Warning
    else:  # BOTH_POLLUTED
        if air_label >= 3 and noise_label >= 3:
            return 3  # Danger
        else:
            return 2  # Warning

# ==================== TÍNH SỐ MẪU CHO TỪNG TRƯỜNG HỢP (3000-5000 mỗi file) ====================

# Mỗi trường hợp có số mẫu ngẫu nhiên từ 3000-5000
scenario_samples = {}
total_samples = 0

for scenario_key, scenario_config in scenarios.items():
    # Random số mẫu cho mỗi trường hợp
    n_samples = random.randint(MIN_SAMPLES_PER_SCENARIO, MAX_SAMPLES_PER_SCENARIO)
    scenario_samples[scenario_key] = n_samples
    total_samples += n_samples

print(f"\nSample distribution (random for each scenario):")
for scenario_key, count in scenario_samples.items():
    print(f"   {scenarios[scenario_key]['name']:20s}: {count:4d} samples")
print(f"\nTotal samples: {total_samples}")

# ==================== TẠO DỮ LIỆU CHO TỪNG TRƯỜNG HỢP ====================

all_dataframes = {}

for scenario_key, scenario_config in scenarios.items():
    n_samples = scenario_samples[scenario_key]
    df = generate_scenario_data(scenario_key, scenario_config, n_samples)
    
    # Thêm labels
    df['AirQuality_Label'] = df.apply(
        lambda row: get_air_quality_label(row['MQ135_AirQuality'], row['MQ7_CO_ppm'], row['PM25_ugm3']),
        axis=1
    )
    df['Noise_Label'] = df['Sound_dB'].apply(get_noise_label)
    df['Alert_Level'] = df.apply(
        lambda row: get_alert_level(row['AirQuality_Label'], row['Noise_Label'], row['Scenario']),
        axis=1
    )
    
    # Thêm features bổ sung
    df['Is_Air_Polluted'] = (df['AirQuality_Label'] >= 2).astype(int)
    df['Is_Noise_Polluted'] = (df['Noise_Label'] >= 2).astype(int)
    
    # Reorder columns
    columns_order = [
        'DateTime', 'Hour', 'DayOfWeek', 'Month',
        'MQ135_AirQuality', 'MQ7_CO_ppm', 'PM25_ugm3', 'Sound_dB',
        'AirQuality_Label', 'Noise_Label', 'Alert_Level',
        'Is_Air_Polluted', 'Is_Noise_Polluted',
        'Scenario', 'Scenario_Name'
    ]
    df = df[columns_order]
    
    all_dataframes[scenario_key] = df
    
    # Lưu file riêng cho từng trường hợp
    filename = f"{output_dir}/dataset_{scenario_key.lower()}.csv"
    df.to_csv(filename, index=False)
    print(f"    [OK] Saved: {filename} ({len(df)} samples)")

# ==================== TẠO FILE TỔNG HỢP ====================

print(f"\n[COMBINED] Creating combined dataset...")
df_combined = pd.concat([all_dataframes[key] for key in scenarios.keys()], ignore_index=True)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

combined_filename = f"{output_dir}/dataset_combined_all.csv"
df_combined.to_csv(combined_filename, index=False)
print(f"    [OK] Saved: {combined_filename} ({len(df_combined)} samples)")

# ==================== THỐNG KÊ ====================

print("\n" + "=" * 70)
print("DATASET STATISTICS")
print("=" * 70)

print(f"\nTotal Samples: {len(df_combined)}")
print(f"\nDistribution by Scenario:")
for scenario_key, scenario_config in scenarios.items():
    count = len(all_dataframes[scenario_key])
    print(f"   {scenario_config['name']:20s}: {count:4d} samples ({count/len(df_combined)*100:.1f}%)")

print(f"\nAlert Level Distribution:")
alert_counts = df_combined['Alert_Level'].value_counts().sort_index()
for level, count in alert_counts.items():
    level_names = {0: 'Very Clean', 1: 'Safe', 2: 'Warning', 3: 'Danger'}
    print(f"   Level {level} ({level_names.get(level, 'Unknown')}): {count:4d} samples ({count/len(df_combined)*100:.1f}%)")

print(f"\nSensor Value Ranges by Scenario:")
for scenario_key, scenario_config in scenarios.items():
    df_scenario = all_dataframes[scenario_key]
    print(f"\n{scenario_config['name']}:")
    print(f"   MQ135:  {df_scenario['MQ135_AirQuality'].min():6.1f} - {df_scenario['MQ135_AirQuality'].max():6.1f} ppm")
    print(f"   CO:     {df_scenario['MQ7_CO_ppm'].min():6.2f} - {df_scenario['MQ7_CO_ppm'].max():6.2f} ppm")
    print(f"   PM2.5:  {df_scenario['PM25_ugm3'].min():6.1f} - {df_scenario['PM25_ugm3'].max():6.1f} µg/m³")
    print(f"   Sound:  {df_scenario['Sound_dB'].min():6.1f} - {df_scenario['Sound_dB'].max():6.1f} dB")

print(f"\nAir Quality Label Distribution:")
print(df_combined['AirQuality_Label'].value_counts().sort_index())

print(f"\nNoise Label Distribution:")
print(df_combined['Noise_Label'].value_counts().sort_index())

print("\n" + "=" * 70)
print("DATASET GENERATION COMPLETE!")
print("=" * 70)
print(f"\nOutput directory: {output_dir}/")
print(f"\nGenerated files:")
print(f"   1. dataset_very_clean.csv      - {len(all_dataframes['VERY_CLEAN'])} samples")
print(f"   2. dataset_safe.csv             - {len(all_dataframes['SAFE'])} samples")
print(f"   3. dataset_air_polluted.csv     - {len(all_dataframes['AIR_POLLUTED'])} samples")
print(f"   4. dataset_noise_polluted.csv   - {len(all_dataframes['NOISE_POLLUTED'])} samples")
print(f"   5. dataset_both_polluted.csv    - {len(all_dataframes['BOTH_POLLUTED'])} samples")
print(f"   6. dataset_combined_all.csv    - {len(df_combined)} samples (Tong hop)")
print("\nTip: Use 'dataset_combined_all.csv' for training the main model.")
print("=" * 70)

