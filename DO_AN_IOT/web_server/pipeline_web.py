"""
Pipeline - Lay du lieu tu Blynk va gui len Web Dashboard
"""

import requests
import time

# ==================== CAU HINH ====================
BLYNK_TOKEN = "x473czYVGJ7ogu1s8I2OFamOHW3evXkX"
BLYNK_URL = f"https://blynk.cloud/external/api/getAll?token={BLYNK_TOKEN}"

# Web Dashboard
WEB_URL = "http://127.0.0.1:5000/api/predict"

# Thoi gian giua cac lan doc (giay) - Real-time
INTERVAL = 1

# ==================== MAIN ====================
def fetch_from_blynk():
    try:
        response = requests.get(BLYNK_URL, timeout=2)  # Timeout nhanh hon
        response.raise_for_status()
        data = response.json()
        
        return {
            'mq135': float(data.get('v0', 0)),
            'mq7': float(data.get('v1', 0)),
            'pm25': float(data.get('v2', 0)),
            'sound': float(data.get('v3', 0))
        }
    except Exception as e:
        print(f"[ERROR] Blynk: {e}")
        return None

def send_to_web(data):
    try:
        response = requests.post(WEB_URL, json=data, timeout=2)  # Timeout nhanh hon
        return response.json()
    except Exception as e:
        print(f"[ERROR] Web: {e}")
        return None

def main():
    print("=" * 50)
    print("PIPELINE: Blynk -> Web Dashboard")
    print("=" * 50)
    print(f"Blynk: {BLYNK_TOKEN[:10]}...")
    print(f"Web: {WEB_URL}")
    print("=" * 50)
    print("\nPress Ctrl+C to stop\n")
    
    while True:
        try:
            # Lay du lieu tu Blynk
            data = fetch_from_blynk()
            
            if data:
                print(f"MQ135: {data['mq135']:.1f} | MQ7: {data['mq7']:.2f} | PM25: {data['pm25']:.1f} | Sound: {data['sound']:.1f}", end="")
                
                # Gui len web
                result = send_to_web(data)
                
                if result and 'alert_text' in result:
                    print(f" -> {result['alert_text']}")
                else:
                    print(" -> Error")
            
            time.sleep(INTERVAL)
            
        except KeyboardInterrupt:
            print("\n\nStopped.")
            break

if __name__ == '__main__':
    main()

