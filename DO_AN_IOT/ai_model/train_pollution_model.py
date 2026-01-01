"""
Train AI Model du doan o nhiem khong khi va am thanh
Input: datasets/dataset_combined_all.csv
Output: pollution_model.pkl + Charts visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Set style cho charts
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Tạo thư mục charts nếu chưa có
charts_dir = 'charts'
if not os.path.exists(charts_dir):
    os.makedirs(charts_dir)
    print(f"[OK] Created directory: {charts_dir}")

print("=" * 70)
print("TRAINING AI MODEL FOR POLLUTION PREDICTION")
print("=" * 70)

# 1. Load data
print("\n[1] Loading dataset...")
df = pd.read_csv('datasets/dataset_combined_all.csv')
print(f"    Loaded {len(df)} samples")

# 2. Feature Engineering
print("\n[2] Creating features...")

# Time features
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['Hour'] = df['DateTime'].dt.hour
df['DayOfWeek'] = df['DateTime'].dt.dayofweek

# Cyclical encoding
df['hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['day_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

# Lag features (previous readings)
for col in ['MQ135_AirQuality', 'MQ7_CO_ppm', 'PM25_ugm3', 'Sound_dB']:
    df[f'{col}_lag1'] = df[col].shift(1)
    df[f'{col}_lag2'] = df[col].shift(2)
    df[f'{col}_rolling_mean'] = df[col].rolling(window=3).mean()

# Drop NaN
df = df.dropna().reset_index(drop=True)
print(f"    After feature engineering: {len(df)} samples")

# 3. Define features and target
FEATURES = [
    'MQ135_AirQuality', 'MQ7_CO_ppm', 'PM25_ugm3', 'Sound_dB',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'MQ135_AirQuality_lag1', 'MQ7_CO_ppm_lag1', 'PM25_ugm3_lag1', 'Sound_dB_lag1',
    'MQ135_AirQuality_lag2', 'MQ7_CO_ppm_lag2', 'PM25_ugm3_lag2', 'Sound_dB_lag2',
    'MQ135_AirQuality_rolling_mean', 'MQ7_CO_ppm_rolling_mean', 
    'PM25_ugm3_rolling_mean', 'Sound_dB_rolling_mean'
]

TARGET = 'Alert_Level'  # 0=Very Clean, 1=Safe, 2=Warning, 3=Danger

X = df[FEATURES]
y = df[TARGET]

print(f"\n[3] Features: {len(FEATURES)}")
print(f"    Target: {TARGET}")
print(f"    Classes: {sorted(y.unique())}")
print(f"    Class distribution:")
print(y.value_counts().sort_index())

# 4. Split data
print("\n[4] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"    Train: {len(X_train)}, Test: {len(X_test)}")

# 5. Train model
print("\n[5] Training CatBoost model...")
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=100
)

# Lưu training history
model.fit(
    X_train, y_train, 
    eval_set=(X_test, y_test), 
    early_stopping_rounds=50
)

# 6. Evaluate
print("\n[6] Evaluating model...")
y_pred = model.predict(X_test)
# Flatten nếu là 2D array
if y_pred.ndim > 1:
    y_pred = y_pred.flatten()
y_pred_proba = model.predict_proba(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n    Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\n    Classification Report:")
class_names = ['Very Clean', 'Safe', 'Warning', 'Danger']
print(classification_report(y_test, y_pred, target_names=class_names))

# 7. Feature importance
print("\n[7] Top 10 Important Features:")
importance = model.get_feature_importance()
feat_imp = pd.DataFrame({'Feature': FEATURES, 'Importance': importance})
feat_imp = feat_imp.sort_values('Importance', ascending=False)
print(feat_imp.head(10).to_string(index=False))

# ==================== TẠO CÁC BIỂU ĐỒ ====================

print("\n[8] Generating visualization charts...")

# Chart 1: Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(f'{charts_dir}/1_confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"    Saved: {charts_dir}/1_confusion_matrix.png")
plt.close()

# Chart 2: Feature Importance
plt.figure(figsize=(12, 8))
top_features = feat_imp.head(15)
plt.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importance', fontsize=12)
plt.title('Top 15 Feature Importance', fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{charts_dir}/2_feature_importance.png', dpi=300, bbox_inches='tight')
print(f"    Saved: {charts_dir}/2_feature_importance.png")
plt.close()

# Chart 3: Class Distribution (Actual vs Predicted)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Actual distribution
actual_counts = pd.Series(y_test).value_counts().sort_index()
ax1.bar([class_names[i] for i in actual_counts.index], actual_counts.values, color='skyblue')
ax1.set_title('Actual Class Distribution (Test Set)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Count', fontsize=12)
ax1.set_xlabel('Class', fontsize=12)
for i, v in enumerate(actual_counts.values):
    ax1.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

# Predicted distribution
pred_counts = pd.Series(y_pred).value_counts().sort_index()
ax2.bar([class_names[i] for i in pred_counts.index], pred_counts.values, color='lightcoral')
ax2.set_title('Predicted Class Distribution (Test Set)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Count', fontsize=12)
ax2.set_xlabel('Class', fontsize=12)
for i, v in enumerate(pred_counts.values):
    ax2.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{charts_dir}/3_class_distribution.png', dpi=300, bbox_inches='tight')
print(f"    Saved: {charts_dir}/3_class_distribution.png")
plt.close()

# Chart 4: Accuracy per Class
class_accuracies = []
for i, class_name in enumerate(class_names):
    if i in y_test.unique():
        mask = y_test == i
        if mask.sum() > 0:
            class_acc = accuracy_score(y_test[mask], y_pred[mask])
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0)
    else:
        class_accuracies.append(0)

plt.figure(figsize=(10, 6))
bars = plt.bar(class_names, class_accuracies, color=['green', 'lightgreen', 'orange', 'red'])
plt.title('Accuracy per Class', fontsize=16, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Class', fontsize=12)
plt.ylim(0, 1.1)
for i, (bar, acc) in enumerate(zip(bars, class_accuracies)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{charts_dir}/4_accuracy_per_class.png', dpi=300, bbox_inches='tight')
print(f"    Saved: {charts_dir}/4_accuracy_per_class.png")
plt.close()

# Chart 5: Prediction Probability Distribution
plt.figure(figsize=(14, 8))
for i, class_name in enumerate(class_names):
    if i < y_pred_proba.shape[1]:
        plt.hist(y_pred_proba[:, i], bins=30, alpha=0.6, label=f'{class_name}', edgecolor='black')
plt.title('Prediction Probability Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Probability', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{charts_dir}/5_prediction_probability.png', dpi=300, bbox_inches='tight')
print(f"    Saved: {charts_dir}/5_prediction_probability.png")
plt.close()

# Chart 6: Training History (nếu có)
try:
    # Lấy training history từ CatBoost
    evals_result = model.get_evals_result()
    if evals_result:
        train_metric = evals_result['learn'].get('MultiClass', [])
        val_metric = evals_result['validation'].get('MultiClass', [])
        
        if train_metric and val_metric:
            plt.figure(figsize=(12, 6))
            iterations = range(len(train_metric))
            plt.plot(iterations, train_metric, label='Train Loss', linewidth=2)
            plt.plot(iterations, val_metric, label='Validation Loss', linewidth=2)
            plt.title('Training History', fontsize=16, fontweight='bold')
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{charts_dir}/6_training_history.png', dpi=300, bbox_inches='tight')
            print(f"    Saved: {charts_dir}/6_training_history.png")
            plt.close()
except:
    print("    Training history not available")

# Chart 7: Sensor Values by Alert Level
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
sensors = ['MQ135_AirQuality', 'MQ7_CO_ppm', 'PM25_ugm3', 'Sound_dB']
sensor_names = ['MQ135 Air Quality (ppm)', 'MQ7 CO (ppm)', 'PM2.5 (µg/m³)', 'Sound (dB)']

for idx, (sensor, sensor_name) in enumerate(zip(sensors, sensor_names)):
    ax = axes[idx // 2, idx % 2]
    data_to_plot = []
    labels_to_plot = []
    
    for class_idx, class_name in enumerate(class_names):
        if class_idx in y_test.unique():
            mask = y_test == class_idx
            if mask.sum() > 0:
                data_to_plot.append(X_test.loc[mask, sensor].values)
                labels_to_plot.append(class_name)
    
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True)
        colors = ['green', 'lightgreen', 'orange', 'red']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(f'{sensor_name} by Alert Level', fontsize=12, fontweight='bold')
        ax.set_ylabel(sensor_name, fontsize=10)
        ax.grid(alpha=0.3)

plt.suptitle('Sensor Values Distribution by Alert Level', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{charts_dir}/7_sensor_by_alert_level.png', dpi=300, bbox_inches='tight')
print(f"    Saved: {charts_dir}/7_sensor_by_alert_level.png")
plt.close()

# Chart 8: Summary Dashboard
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Overall accuracy
ax1 = fig.add_subplot(gs[0, 0])
ax1.text(0.5, 0.5, f'{accuracy*100:.2f}%', ha='center', va='center', 
         fontsize=48, fontweight='bold', color='steelblue')
ax1.text(0.5, 0.2, 'Overall Accuracy', ha='center', va='center', 
         fontsize=14, fontweight='bold')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')

# Class distribution pie
ax2 = fig.add_subplot(gs[0, 1:])
actual_counts = pd.Series(y_test).value_counts().sort_index()
colors_pie = ['green', 'lightgreen', 'orange', 'red']
ax2.pie(actual_counts.values, labels=[class_names[i] for i in actual_counts.index], 
        autopct='%1.1f%%', startangle=90, colors=colors_pie[:len(actual_counts)])
ax2.set_title('Test Set Class Distribution', fontsize=12, fontweight='bold')

# Top 5 features
ax3 = fig.add_subplot(gs[1, :])
top5 = feat_imp.head(5)
ax3.barh(range(len(top5)), top5['Importance'], color='steelblue')
ax3.set_yticks(range(len(top5)))
ax3.set_yticklabels(top5['Feature'])
ax3.set_xlabel('Importance', fontsize=10)
ax3.set_title('Top 5 Most Important Features', fontsize=12, fontweight='bold')
ax3.invert_yaxis()

# Confusion matrix small
ax4 = fig.add_subplot(gs[2, :2])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=class_names, yticklabels=class_names, cbar_kws={'shrink': 0.8})
ax4.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
ax4.set_ylabel('True Label', fontsize=10)
ax4.set_xlabel('Predicted Label', fontsize=10)

# Class accuracies
ax5 = fig.add_subplot(gs[2, 2])
bars = ax5.barh(class_names, class_accuracies, color=['green', 'lightgreen', 'orange', 'red'])
ax5.set_xlabel('Accuracy', fontsize=10)
ax5.set_title('Accuracy per Class', fontsize=12, fontweight='bold')
ax5.set_xlim(0, 1.1)
for i, (bar, acc) in enumerate(zip(bars, class_accuracies)):
    ax5.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
             f'{acc:.3f}', va='center', fontweight='bold', fontsize=9)

plt.suptitle('Model Training Summary Dashboard', fontsize=18, fontweight='bold', y=0.98)
plt.savefig(f'{charts_dir}/8_summary_dashboard.png', dpi=300, bbox_inches='tight')
print(f"    Saved: {charts_dir}/8_summary_dashboard.png")
plt.close()

# 8. Save model
print("\n[9] Saving model...")
joblib.dump(model, 'pollution_model.pkl')
joblib.dump(FEATURES, 'model_features.pkl')
print("    Saved: pollution_model.pkl")
print("    Saved: model_features.pkl")

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"\nAll charts saved in: {charts_dir}/")
print(f"   - 1_confusion_matrix.png")
print(f"   - 2_feature_importance.png")
print(f"   - 3_class_distribution.png")
print(f"   - 4_accuracy_per_class.png")
print(f"   - 5_prediction_probability.png")
print(f"   - 6_training_history.png (if available)")
print(f"   - 7_sensor_by_alert_level.png")
print(f"   - 8_summary_dashboard.png")
print("=" * 70)
