"""
Web Dashboard - Hien thi du lieu o nhiem real-time
"""

from flask import Flask, render_template, jsonify, request
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import json

app = Flask(__name__)

# Load model
import os
model_path = os.path.join(os.path.dirname(__file__), '..', 'ai_model', 'pollution_model.pkl')
features_path = os.path.join(os.path.dirname(__file__), '..', 'ai_model', 'model_features.pkl')

print("Loading model...")
print(f"Model path: {model_path}")
model = joblib.load(model_path)
FEATURES = joblib.load(features_path)
print("Model loaded!")

# Luu tru du lieu
history = []
MAX_HISTORY = 100  # Giu 100 records gan nhat

def get_alert_info(level):
    """Model mới có 4 levels: 0=Very Clean, 1=Safe, 2=Warning, 3=Danger"""
    if level == 0:
        return {"text": "TRONG LANH", "color": "#2ecc71", "icon": "leaf"}
    elif level == 1:
        return {"text": "AN TOAN", "color": "#27ae60", "icon": "check-circle"}
    elif level == 2:
        return {"text": "CANH BAO", "color": "#f39c12", "icon": "exclamation-triangle"}
    else:  # level == 3
        return {"text": "NGUY HIEM", "color": "#e74c3c", "icon": "times-circle"}

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/data')
def get_data():
    """Tra ve du lieu moi nhat"""
    if history:
        return jsonify(history[-1])
    return jsonify({})

@app.route('/api/history')
def get_history():
    """Tra ve lich su du lieu"""
    return jsonify(history[-50:])  # 50 records gan nhat

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        mq135 = float(data.get('mq135', 0))
        mq7 = float(data.get('mq7', 0))
        pm25 = float(data.get('pm25', 0))
        sound = float(data.get('sound', 0))
        
        now = datetime.now()
        hour = now.hour
        dow = now.weekday()
        
        # Tao features
        features = {
            'MQ135_AirQuality': mq135,
            'MQ7_CO_ppm': mq7,
            'PM25_ugm3': pm25,
            'Sound_dB': sound,
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'day_sin': np.sin(2 * np.pi * dow / 7),
            'day_cos': np.cos(2 * np.pi * dow / 7),
        }
        
        # Lag features
        if len(history) >= 2:
            features['MQ135_AirQuality_lag1'] = history[-1]['mq135']
            features['MQ7_CO_ppm_lag1'] = history[-1]['mq7']
            features['PM25_ugm3_lag1'] = history[-1]['pm25']
            features['Sound_dB_lag1'] = history[-1]['sound']
        else:
            features['MQ135_AirQuality_lag1'] = mq135
            features['MQ7_CO_ppm_lag1'] = mq7
            features['PM25_ugm3_lag1'] = pm25
            features['Sound_dB_lag1'] = sound
        
        if len(history) >= 3:
            features['MQ135_AirQuality_lag2'] = history[-2]['mq135']
            features['MQ7_CO_ppm_lag2'] = history[-2]['mq7']
            features['PM25_ugm3_lag2'] = history[-2]['pm25']
            features['Sound_dB_lag2'] = history[-2]['sound']
        else:
            features['MQ135_AirQuality_lag2'] = mq135
            features['MQ7_CO_ppm_lag2'] = mq7
            features['PM25_ugm3_lag2'] = pm25
            features['Sound_dB_lag2'] = sound
        
        # Rolling mean
        if len(history) >= 3:
            features['MQ135_AirQuality_rolling_mean'] = np.mean([h['mq135'] for h in history[-3:]])
            features['MQ7_CO_ppm_rolling_mean'] = np.mean([h['mq7'] for h in history[-3:]])
            features['PM25_ugm3_rolling_mean'] = np.mean([h['pm25'] for h in history[-3:]])
            features['Sound_dB_rolling_mean'] = np.mean([h['sound'] for h in history[-3:]])
        else:
            features['MQ135_AirQuality_rolling_mean'] = mq135
            features['MQ7_CO_ppm_rolling_mean'] = mq7
            features['PM25_ugm3_rolling_mean'] = pm25
            features['Sound_dB_rolling_mean'] = sound
        
        # Predict với AI Model (ưu tiên AI)
        X = pd.DataFrame([features])[FEATURES]
        pred_result = model.predict(X)
        # Flatten nếu là 2D array
        if pred_result.ndim > 1:
            pred_result = pred_result.flatten()
        ai_prediction = int(pred_result[0])
        proba_result = model.predict_proba(X)
        probabilities = [float(p) for p in proba_result[0]]  # 4 probabilities cho 4 classes
        
        # FALLBACK: Kiểm tra giá trị cảm biến trước AI (ưu tiên fallback)
        critical_air = (mq7 > 35 or pm25 > 150 or mq135 > 200)
        critical_noise = (sound > 85)
        warning_air = (mq7 > 9 or pm25 > 75 or mq135 > 100)
        warning_noise = (sound > 65)
        
        # Logic: Fallback > AI (đảm bảo đồng bộ)
        if critical_air or critical_noise:
            prediction = 3
            probabilities = [0.0, 0.0, 0.0, 1.0]
        elif warning_air or warning_noise:
            if ai_prediction < 2:
                prediction = 2
                probabilities = [0.0, 0.0, 1.0, 0.0]
            else:
                prediction = ai_prediction
        else:
            prediction = ai_prediction
        
        alert_info = get_alert_info(prediction)
        
        # Luu vao history
        record = {
            'timestamp': now.strftime('%H:%M:%S'),
            'datetime': now.strftime('%Y-%m-%d %H:%M:%S'),
            'mq135': mq135,
            'mq7': mq7,
            'pm25': pm25,
            'sound': sound,
            'alert_level': prediction,
            'alert_text': alert_info['text'],
            'alert_color': alert_info['color'],
            'prob_very_clean': round(probabilities[0] * 100, 1) if len(probabilities) > 0 else 0.0,
            'prob_safe': round(probabilities[1] * 100, 1) if len(probabilities) > 1 else 0.0,
            'prob_warning': round(probabilities[2] * 100, 1) if len(probabilities) > 2 else 0.0,
            'prob_danger': round(probabilities[3] * 100, 1) if len(probabilities) > 3 else 0.0
        }
        
        history.append(record)
        if len(history) > MAX_HISTORY:
            history.pop(0)
        
        return jsonify(record)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("WEB DASHBOARD - POLLUTION MONITOR")
    print("=" * 50)
    print("Open browser: http://localhost:5000")
    print("=" * 50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

