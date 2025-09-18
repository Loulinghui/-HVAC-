import paho.mqtt.client as mqtt
from paho.mqtt.client import CallbackAPIVersion
import json
import time
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import threading
import requests
from datetime import datetime, timezone
import math
import pvlib
# 引入绘图库
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties

# --- 0. 环境与常量设置 ---
warnings.filterwarnings("ignore")

# --- MQTT 配置 ---
MQTT_BROKER = "47.105.58.2"
MQTT_PORT = 1883
CLIENT_ID = f"python-prediction-monitor-final-{int(time.time())}"
PROJECT_ID = "240923"
DEVICE_ID = "YCM052-0000-5155"
DEVICE_LATITUDE = 32.9947
DEVICE_LONGITUDE = 112.5325

TOPIC_TO_SUBSCRIBE = f"/topic/{PROJECT_ID}/topic_dev2ser"
TOPIC_TO_PUBLISH = f"/topic/topic_ser2dev/{DEVICE_ID}"

# --- 模型配置 ---
# 【重要】请确保此路径指向您存放训练好的模型文件的文件夹
MODEL_DIR = "/Users/loulinghui/Desktop/gemini/8.15模拟"
TIME_STEPS = 6
REAL_DATA_PATH = "hvac_data_collection.csv" # 用于预热的历史数据

# --- 天气 API 配置 ---
HEFENG_KEY = "a9d02e534941434abee39c53ba260de7"
OPENWEATHER_KEY = "5ce26697285b823131dd1df18c1bf03c"

# --- 1. 全局数据存储 ---
DEVICE_DATA = {}
LAST_PREDICTION_TIME = {}
DEVICE_STATUS = { "online": False, "last_seen": 0 }
STATUS_TIMEOUT = 360 # 6分钟
PREDICTION_HISTORY = [] # 用于存储绘图数据（包含风速信息）

# 智能风速残差修正系统
SPEED_BASED_EMA_RESIDUAL = {}  # {devid: {1: residual, 2: residual, 3: residual}}
LAST_FAN_SPEED = {}           # {devid: fan_speed}
SPEED_SWITCH_COUNT = {}       # {devid: switch_count}
LAST_PRED = {}                # 保留但改为 {devid: {'predicted_temp': xxx, 'fan_speed': xxx}}
LAST_MODEL_CHANGE = {}        # 保留


ROOM_CONFIG = {
    '财务室': {'面积': 23.21, "最大总冷量": 827, "最大显冷量": 599 ,'风盘型号': 'FP-136', '制冷高系数': 5.99, '制冷中系数': 5, '制冷低系数': 3.59, '制热高系数': 8.42, '制热中系数': 6.82, '制热低系数': 4.8},
    '董事长办公室': {'面积': 40.71, "最大总冷量":827 , "最大显冷量": 599,'风盘型号': 'FP-136', '制冷高系数': 5.99, '制冷中系数': 5, '制冷低系数': 3.59, '制热高系数': 8.42, '制热中系数': 6.82, '制热低系数': 4.8},
    '南区总室': {'面积': 12.08, "最大总冷量": 312, "最大显冷量": 223,'风盘型号': 'FP-51', '制冷高系数': 2.23, '制冷中系数': 1.82, '制冷低系数': 1.35, '制热高系数': 3.23, '制热中系数': 2.72, '制热低系数': 2.01},
    '软件办公室': {'面积': 38.52, "最大总冷量": 519, "最大显冷量": 371,'风盘型号': 'FP-85', '制冷高系数': 3.71, '制冷中系数': 3.03, '制冷低系数': 2.2, '制热高系数': 5.17, '制热中系数': 4.18, '制热低系数': 3.00},
    '设计部经理办公室': {'面积': 9.91, "最大总冷量": 312, "最大显冷量": 223,'风盘型号': 'FP-51', '制冷高系数': 2.23, '制冷中系数': 1.82, '制冷低系数': 1.35, '制热高系数': 3.23, '制热中系数': 2.72, '制热低系数': 2.01},
    '市场部公共办公室': {'面积': 108.43,"最大总冷量": 595, "最大显冷量": 428, '风盘型号': 'FP-102', '制冷高系数': 4.28, '制冷中系数': 3.58, '制冷低系数': 2.58, '制热高系数': 6.28, '制热中系数': 5.08, '制热低系数': 3.51},
    '市场部经理办公室': {'面积': 20.95, "最大总冷量": 827, "最大显冷量": 599,'风盘型号': 'FP-136', '制冷高系数': 5.99, '制冷中系数': 5, '制冷低系数': 3.59, '制热高系数': 8.42, '制热中系数': 6.82, '制热低系数': 4.8},
    '四楼接待室': {'面积': 32.29, "最大总冷量": 595, "最大显冷量": 428,'风盘型号': 'FP-102', '制冷高系数': 4.28, '制冷中系数': 3.58, '制冷低系数': 2.58, '制热高系数': 6.28, '制热中系数': 5.08, '制热低系数': 3.51},
    '行政人事部': {'面积': 34.59, "最大总冷量": 827, "最大显冷量": 599,'风盘型号': 'FP-136', '制冷高系数': 5.99, '制冷中系数': 5, '制冷低系数': 3.59, '制热高系数': 8.42, '制热中系数': 6.82, '制热低系数': 4.8},
    '小会议室': {'面积': 32.57, "最大总冷量": 519, "最大显冷量":371,'风盘型号': 'FP-85', '制冷高系数': 3.71, '制冷中系数': 3.03, '制冷低系数': 2.2, '制热高系数': 5.17, '制热中系数': 4.18, '制热低系数': 3.00},
    '生产部经理室': {'面积': 32.68, "最大总冷量": 312, "最大显冷量": 223,'风盘型号': 'FP-51', '制冷高系数': 2.23, '制冷中系数': 1.82, '制冷低系数': 1.35, '制热高系数': 3.23, '制热中系数': 2.72, '制热低系数': 2.01},
    '仓管部': {'面积': 30.35, "最大总冷量": 595, "最大显冷量": 428,'风盘型号': 'FP-102', '制冷高系数': 4.28, '制冷中系数': 3.58, '制冷低系数': 2.58, '制热高系数': 6.28, '制热中系数': 5.08, '制热低系数': 3.51},
    '研发部': {'面积': 45.64, "最大总冷量": 595, "最大显冷量": 428,'风盘型号': 'FP-102', '制冷高系数': 4.28, '制冷中系数': 3.58, '制冷低系数': 2.58, '制热高系数': 6.28, '制热中系数': 5.08, '制热低系数': 3.51},
    '成品仓库': {'面积': 80.49,"最大总冷量": 934, "最大显冷量": 686, '风盘型号': 'FP-170', '制冷高系数': 6.86, '制冷中系数': 6.05, '制冷低系数': 4.58, '制热高系数': 9.84, '制热中系数': 8.56, '制热低系数': 6.40},
    '测试间': {'面积': 48.48,"最大总冷量": 1124, "最大显冷量": 866, '风盘型号': 'FP-204', '制冷高系数': 8.66, '制冷中系数': 7.21, '制冷低系数': 4.94, '制热高系数': 11.84, '制热中系数': 9.71, '制热低系数': 6.27}
}



PHYSICS_CONFIG = {
    "INSULATION_FACTOR": 0.008, "HEAT_GAIN_EFFECT_FACTOR": 0.003, "SOLAR_HEAT_GAIN_FACTOR": 0.8, "AC_COOLING_EFFICIENCY": 0.25, "AC_HEATING_EFFICIENCY": 0.20, "EVENT_PROBABILITY": 0.05, "THERMAL_MASS_FACTOR": 0.7, "OUTDOOR_INFILTRATION": 0.15,
    "WATTS_TO_TEMP_CHANGE_FACTOR": 300 / (1.2 * 2.8 * 1005),
    "WATTS_TO_HUMIDITY_CHANGE_FACTOR": 0.0005, 
    "FAN_SPEED_U_X_MULTIPLIERS": {
        3: 1.0, 2: 0.75, 1: 0.50, 0: 0.0
    },
}


# --- 智能风速调节下的残差修正系统 ---

def initialize_speed_residuals(devid):
    """初始化设备的分档残差存储"""
    if devid not in SPEED_BASED_EMA_RESIDUAL:
        SPEED_BASED_EMA_RESIDUAL[devid] = {1: 0.0, 2: 0.0, 3: 0.0}
    if devid not in LAST_FAN_SPEED:
        LAST_FAN_SPEED[devid] = 3  # 默认3档
    if devid not in SPEED_SWITCH_COUNT:
        SPEED_SWITCH_COUNT[devid] = 0

# =============================================================================
# --- 请将此函数完整替换掉脚本中的旧版本 ---
# =============================================================================
def update_smart_residual(devid, current_temp, last_prediction, current_fan_speed, last_fan_speed):
    """
    【全新修正版】智能风速调节下的残差修正，引入“误差渗透”机制
    """
    initialize_speed_residuals(devid)
    
    current_residual = current_temp - last_prediction
    fan_speed_changed = current_fan_speed != last_fan_speed
    
    print(f"   -> [智能残差修正] 当前误差: {current_residual:+.2f}°C")
    print(f"   -> [风速状态] 上次: {last_fan_speed}档 -> 当前: {current_fan_speed}档")
    
    if fan_speed_changed:
        SPEED_SWITCH_COUNT[devid] += 1
        print(f"   -> [风速切换] 检测到风速变化，累计切换次数: {SPEED_SWITCH_COUNT[devid]}")
    
    # 极端误差处理
    if abs(current_residual) > 2.0:
        print(f"   -> [极端误差] 检测到异常误差({current_residual:+.2f}°C)，本次不更新所有残差")
        return SPEED_BASED_EMA_RESIDUAL[devid].get(current_fan_speed, 0.0), fan_speed_changed

    # --- 全新残差更新逻辑 ---
    base_alpha = 0.66
    bleed_alpha = 0.25 # 定义一个较低的“渗透”学习率

    # 1. 更新当前正在运行的风速档位
    active_alpha = base_alpha
    if fan_speed_changed:
        active_alpha *= 0.3 # 风速切换后，对当前档位的学习也保守一些
    
    old_residual_active = SPEED_BASED_EMA_RESIDUAL[devid][current_fan_speed]
    new_residual_active = active_alpha * current_residual + (1 - active_alpha) * old_residual_active
    SPEED_BASED_EMA_RESIDUAL[devid][current_fan_speed] = new_residual_active
    print(f"   -> [风速{current_fan_speed}档-动态更新] α={active_alpha:.2f}, 旧值:{old_residual_active:+.3f} -> 新值:{new_residual_active:+.3f}")

    # 2. 【核心修正】将当前误差信息，“渗透”给其他未运行的档位
    print(f"   -> [误差渗透] 将误差 {current_residual:+.2f}°C 以低学习率(α={bleed_alpha})同步给其他档位...")
    for speed in [1, 2, 3]:
        if speed != current_fan_speed:
            old_residual_inactive = SPEED_BASED_EMA_RESIDUAL[devid][speed]
            # 用一个较低的学习率，让非当前档位也能“借鉴”最新的误差信息
            new_residual_inactive = bleed_alpha * current_residual + (1 - bleed_alpha) * old_residual_inactive
            SPEED_BASED_EMA_RESIDUAL[devid][speed] = new_residual_inactive
    
    # 更新上次风速记录
    LAST_FAN_SPEED[devid] = current_fan_speed
    
    # 显示所有风速档位的残差状态
    print(f"   -> [全档残差状态] 1档:{SPEED_BASED_EMA_RESIDUAL[devid][1]:+.3f}, "
          f"2档:{SPEED_BASED_EMA_RESIDUAL[devid][2]:+.3f}, "
          f"3档:{SPEED_BASED_EMA_RESIDUAL[devid][3]:+.3f}")
    
    return new_residual_active, fan_speed_changed


def get_residual_for_prediction(devid, target_fan_speed):
    """
    获取指定风速档位的残差修正值用于预测
    
    Args:
        devid: 设备ID  
        target_fan_speed: 目标风速档位
    
    Returns:
        float: 该风速档位的残差修正值
    """
    initialize_speed_residuals(devid)
    
    base_residual = SPEED_BASED_EMA_RESIDUAL[devid][target_fan_speed]
    current_fan_speed = LAST_FAN_SPEED[devid]
    
    # 如果预测的风速与当前风速不同，需要调整残差权重
    if target_fan_speed != current_fan_speed:
        # 跨风速档预测时，降低残差修正的影响
        adjusted_residual = base_residual * 0.5
        print(f"   -> [跨档预测] {current_fan_speed}档->{target_fan_speed}档，残差调整: "
              f"{base_residual:+.3f} * 0.5 = {adjusted_residual:+.3f}")
        return adjusted_residual
    
    return base_residual

def analyze_residual_performance(devid):
    """分析残差修正的整体表现"""
    if devid not in SPEED_BASED_EMA_RESIDUAL:
        return
    
    residuals = SPEED_BASED_EMA_RESIDUAL[devid]
    switch_count = SPEED_SWITCH_COUNT.get(devid, 0)
    
    print(f"\n--- 残差修正性能分析 (设备: {devid}) ---")
    print(f"风速切换次数: {switch_count}")
    
    for speed in [1, 2, 3]:
        residual = residuals[speed]
        status = "优秀" if abs(residual) < 0.2 else "良好" if abs(residual) < 0.5 else "需改进"
        print(f"{speed}档残差: {residual:+.3f}°C ({status})")
    
    # 计算整体残差水平
    avg_abs_residual = sum(abs(r) for r in residuals.values()) / 3
    overall_status = "优秀" if avg_abs_residual < 0.2 else "良好" if avg_abs_residual < 0.5 else "需改进"
    print(f"平均绝对残差: {avg_abs_residual:.3f}°C ({overall_status})")
    
    return {
        'residuals': residuals.copy(),
        'switch_count': switch_count,
        'avg_abs_residual': avg_abs_residual,
        'status': overall_status
    }


# --- 2. 特征工程 (与训练脚本完全一致) ---
def feature_engineering_advanced(df, room_config, physics_config):
    """【黄金版】特征工程，新增“有效制冷潜力”特征"""
    print("   -> 正在运行特征工程 (黄金版)...")
    df = df.copy()
    df['时间'] = pd.to_datetime(df['时间'])
    df['小时'], df['星期几'], df['月'] = df['时间'].dt.hour, df['时间'].dt.dayofweek, df['时间'].dt.month
    df['温差'], df['室内外温差'] = df['室内温度'] - df['设置温度'], df['室内温度'] - df['室外温度']
    df['室内外湿度差'] = df['室内湿度'] - df['室外湿度']
    df['设备负荷'] = df['开机'] * df['风速']
    df['是否工作时间'] = ((df['小时'] >= 8) & (df['小时'] < 18)).astype(int)
    
    # 临时的WaterSupplyManager，只为调用效率计算方法
    # 注意：这里我们不再需要传入完整的water_config，因为它只用于初始化我们不用的部分
    class TempWaterManager:
        def get_cooling_efficiency_factor(self, supply_temp):
            if supply_temp <= 14.0: return min(7.0 / supply_temp if supply_temp > 0 else 0, 1.2)
            else: return max(0, -0.0625 * supply_temp + 1.375)
    
    water_manager = TempWaterManager()
    df['g_t_efficiency'] = df['供水温度'].apply(water_manager.get_cooling_efficiency_factor)
    u_x_map = physics_config["FAN_SPEED_U_X_MULTIPLIERS"]
    df['u_x_efficiency'] = df['风速'].map(u_x_map).fillna(0)
    
    def get_max_sensible_cooling(room_name):
        return room_config.get(str(room_name), {}).get('最大显冷量', 0)

    df['max_sensible_cooling'] = df['名称'].apply(get_max_sensible_cooling)
    df['有效制冷潜力'] = df['max_sensible_cooling'] * df['g_t_efficiency'] * df['u_x_efficiency']
    
    for lag in [1, 2, 3]: df[f'室内温度_lag_{lag}'] = df['室内温度'].shift(lag)
    for window in [3, 6]:
        df[f'室内温度_ma_{window}'] = df['室内温度'].rolling(window, min_periods=1).mean()
        df[f'温差_ma_{window}'] = df['温差'].rolling(window, min_periods=1).mean()
        
    df = df.drop(columns=['g_t_efficiency', 'u_x_efficiency', 'max_sensible_cooling'])
    return df


# --- 3. 天气模块 (带备用API) ---
def get_weather_hefeng(lat, lon, key):
    url = f"https://api.qweather.com/v7/weather/now?location={lon:.2f},{lat:.2f}&key={key}"
    try:
        r = requests.get(url, timeout=4).json()
        if r.get('code') == '200':
            now = r['now']
            return float(now['temp']), float(now['humidity']), float(now.get('cloud', 50))
    except Exception as e:
        print(f"   -> [和风天气] API请求失败: {e}")
    return None

def get_weather_openweathermap(lat, lon, key):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={key}&units=metric"
    try:
        r = requests.get(url, timeout=4).json()
        if r.get('cod') == 200:
            return r['main']['temp'], r['main']['humidity'], r['clouds']['all']
    except Exception as e:
        print(f"   -> [OpenWeatherMap] API请求失败: {e}")
    return None

def calc_solar_radiation(lat, lon, cloud):
    now = datetime.now(timezone.utc)
    times = pd.DatetimeIndex([now])
    solpos = pvlib.solarposition.get_solarposition(times, lat, lon)
    zenith = solpos['zenith'].iloc[0]
    if zenith > 90: return 0
    n = now.timetuple().tm_yday
    I0 = 1367 * (1 + 0.033 * math.cos(2 * math.pi * n / 365))
    ghi_clear = I0 * math.cos(math.radians(zenith)) * 0.75
    ghi_clear = max(ghi_clear, 0)
    return ghi_clear * (1 - 0.75 * (cloud/100.0)**3.4)

def get_outdoor_conditions(lat, lon):
    print("--- 正在获取实时室外天气数据... ---")
    weather_data = get_weather_hefeng(lat, lon, HEFENG_KEY)
    if weather_data:
        temp, humidity, cloud = weather_data
        radiation = calc_solar_radiation(lat, lon, cloud)
        print(f"   -> [和风天气] 获取成功: 温度={temp}°C, 湿度={humidity}%, 光照={radiation:.2f} W/m²")
        return temp, humidity, radiation
    
    print("   -> [和风天气] 获取失败，正在尝试备用API...")
    weather_data_owm = get_weather_openweathermap(lat, lon, OPENWEATHER_KEY)
    if weather_data_owm:
        temp, humidity, cloud = weather_data_owm
        radiation = calc_solar_radiation(lat, lon, cloud)
        print(f"   -> [OpenWeatherMap] 获取成功: 温度={temp}°C, 湿度={humidity}%, 光照={radiation:.2f} W/m²")
        return temp, humidity, radiation

    print("   -> 🚨 所有天气API均获取失败。")
    return np.nan, np.nan, np.nan

# --- 4. 预测核心函数 ---

def make_prediction(data_window_raw, model, feature_names, room_config, physics_config):
    """【黄金版】调用新的特征工程函数"""
    print("   -> 正在为模型构建特征...")
    # 【修改】调用新的特征工程，并传入配置
    df_featured = feature_engineering_advanced(data_window_raw.copy(), room_config, physics_config)
    
    df_featured = df_featured.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    for col in feature_names:
        if col not in df_featured.columns:
            df_featured[col] = 0
            
    X = df_featured[feature_names]
    prediction = model.predict(X.tail(1))
    
    return prediction[0]


# =============================================================================
# --- 方案B: 严格按照您提出的规则修正 ---
# =============================================================================


def predict_with_different_fan_speeds(df_window, model, feature_names, current_state, room_config, physics_config):
    """【黄金版】调用新的make_prediction，并加入物理规则强制校正"""
    predictions = {}
    print("   -> 正在预测不同风速下的温度变化...")
    
    for fan_speed in [1, 2, 3]:
        test_window = df_window.copy()
        test_window.loc[test_window.index[-1], '风速'] = fan_speed
        
        # 【修改】调用新的make_prediction，并传入配置
        predicted_change = make_prediction(test_window, model, feature_names, room_config, physics_config)
        predictions[fan_speed] = predicted_change
        print(f"      风速{fan_speed}档 -> 模型原始预测变化量: {predicted_change:+.4f}°C")
    
    # --- 物理规则强制校正 ---
    corrected_predictions = predictions.copy()
    correction_multipliers = {2: 1.15, 3: 1.1} 
    if corrected_predictions[2] > corrected_predictions[1]:
        corrected_predictions[2] = corrected_predictions[1] * correction_multipliers[2]
    if corrected_predictions[3] > corrected_predictions[2]:
        corrected_predictions[3] = corrected_predictions[2] * correction_multipliers[3]
    if corrected_predictions[2] < corrected_predictions[1] and corrected_predictions[2] < 0 and corrected_predictions[1] < 0: # 修正降温时的小bug
         if corrected_predictions[2] > corrected_predictions[1]: corrected_predictions[2] = corrected_predictions[1]

    if predictions != corrected_predictions:
        print("   -> 经过物理规则校正后的预测变化量:")
        for speed in [1, 2, 3]:
            print(f"      风速{speed}档 -> 最终采纳变化量: {corrected_predictions[speed]:+.4f}°C")
    else:
        print("   -> 模型原始预测符合物理规则，无需校正。")

    return corrected_predictions


# =============================================================================
# --- 请将此函数完整替换掉脚本中的旧版本 ---
# =============================================================================
# =============================================================================
# --- 请将此函数完整替换掉脚本中的旧版本 ---
# =============================================================================
def decide_optimal_fan_speed(current_temp, set_temp, fan_speed_predictions, devid, current_fan_speed):
    """【最终黄金版】决策逻辑，将“控制意图”作为最高优先级"""
    print("   -> 正在分析最优风速...")
    
    # 1. 计算所有选项的最终预测温度
    final_predictions = {}
    for fan_speed in [1, 2, 3]:
        # 注意：这里我们直接使用校正后的 fan_speed_predictions
        change = fan_speed_predictions.get(fan_speed, 0)
        speed_residual = get_residual_for_prediction(devid, fan_speed)
        final_temp = current_temp + change + speed_residual
        final_predictions[fan_speed] = final_temp
        print(f"      风速{fan_speed}档 -> 校正后变化量: {change:+.3f}, 残差修正: {speed_residual:+.3f} -> 最终预测温度: {final_temp:.2f}°C")

    # --- 全新决策逻辑：基于“控制意图” ---
    best_fan_speed = current_fan_speed # 默认保持当前风速
    comfort_zone_upper = set_temp + 0.5

    # 2. 判断当前的主要需求：是需要降温，还是已经达标
    if current_temp > comfort_zone_upper:
        # --- 主要需求：需要降温 ---
        print(f"   -> [决策分析] 当前温度({current_temp:.1f}°C)高于舒适区，首要目标是【降温】。")
        
        # 【核心规则】在所有选项中，找到那个能让未来温度变得最低的选项
        # min(字典, key=字典.get) 会返回拥有最小值的那个键
        best_fan_speed = min(final_predictions, key=final_predictions.get)
        
        print(f"   -> 🎯 决策：为实现最大降温效果，选择预测温度最低的风速 {best_fan_speed} 档。")
            
    else:
        # --- 主要需求：已在舒适区或过冷，优先考虑【节能】 ---
        print(f"   -> [决策分析] 当前温度({current_temp:.1f}°C)在舒适区内或更低，首要目标是【节能】。")
        comfort_zone_lower = set_temp - 0.5
        
        # 筛选出所有能将温度维持在舒适区的选项
        options_in_comfort_zone = {}
        for fan_speed, predicted_temp in final_predictions.items():
             if comfort_zone_lower <= predicted_temp <= comfort_zone_upper + 0.2: # 稍微放宽上限
                options_in_comfort_zone[fan_speed] = predicted_temp

        if options_in_comfort_zone:
            # 如果有达标选项，选择最节能的（风速最低的）
            best_fan_speed = min(options_in_comfort_zone.keys())
            print(f"   -> 🎯 决策：选择最节能的风速 {best_fan_speed} 档，可将温度维持在舒适区。")
        else:
            # 如果所有选项都会导致过冷，选择影响最小的（预测温度最高的）
            best_fan_speed = max(final_predictions, key=final_predictions.get)
            print(f"   -> 🎯 决策：所有档位都会导致过冷，选择影响最小的风速 {best_fan_speed} 档。")

    return best_fan_speed, final_predictions[best_fan_speed]



def send_command_305(client, fan_speed, device_id):
    """发送风速调节指令"""
    print(f"\n[发送指令 305] -> 向设备 {device_id} 设置新风速...")
    
    latest_data = DEVICE_DATA.get(device_id, [])[-1]
    power_state = 1  # 保持开机状态
    mode = latest_data.get('运行模式', 2)
    set_temp = latest_data.get('设置温度', 24.5)
    lock = latest_data.get('锁定', 0)
    
    payload = {
        "ver": "1.0.2", 
        "devid": device_id, 
        "cmd": "305",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": {
            "yc04": int(power_state), 
            "yc06": int(mode), 
            "yc05": int(fan_speed), 
            "yc35": int(lock), 
            "yc09": int(float(set_temp) * 10)
        }
    }
    
    client.publish(TOPIC_TO_PUBLISH, json.dumps(payload))
    print(f"指令 305 已发送: 空调=开, 风速={fan_speed}档")

# --- 6. MQTT 回调函数 ---
def on_connect(client, userdata, flags, rc, properties):
    if rc == 0:
        print("✅ 成功连接到 MQTT 服务器!")
        client.subscribe(TOPIC_TO_SUBSCRIBE)
    else:
        print(f"❌ 连接失败，返回码: {rc}")

# =============================================================================
# --- 请将此函数完整替换掉脚本中的旧版本 ---
# =============================================================================
def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        devid = data.get("devid")
        if devid != DEVICE_ID: return
        
        DEVICE_STATUS["online"] = True
        DEVICE_STATUS["last_seen"] = time.time()
        
        cmd = data.get("cmd")

        # 【【【***核心修改：增加对 305 指令回复的处理***】】】
        if cmd == "305":
            if data.get("msg") == "success":
                print(f"✅ [指令确认] 设备 {devid} 已成功执行修改风速。")
            else:
                print(f"🚨 [指令失败] 设备 {devid} 未能成功执行调节指令！设备返回: {data}")
            return # 处理完指令回执后，直接返回，不再执行后续操作

        # 如果不是305指令，再继续判断是否是数据上报指令
        if cmd not in ["307", "2022"]:
            return
            
        mqtt_data = data.get("data", {})
        if not mqtt_data: return

        current_state = {
            "时间": pd.to_datetime(data.get("time", time.strftime("%Y-%m-%d %H:%M:%S"))),
            "名称": "软件办公室", # 【【【***请在这里加入这一行***】】
            "室内温度": float(mqtt_data.get("yc12", 0)) / 10.0,
            "设置温度": float(mqtt_data.get("yc09", 0)) / 10.0,
            "风速": int(mqtt_data.get("yc05", 0)),
            "开机": int(mqtt_data.get("yc04", 0)),
            "室内湿度": float(mqtt_data.get("yc66", 0)) / 10.0,
            "供水温度": float(mqtt_data.get("yc10", 0)) / 10.0,
            "回水温度": float(mqtt_data.get("yc11", 0)) / 10.0,
            "运行模式": int(mqtt_data.get("yc06", 2)),
            "锁定": int(mqtt_data.get("yc35", 0))
        }
        
        if devid in DEVICE_DATA and len(DEVICE_DATA[devid]) > 0:
            last_timestamp = pd.to_datetime(DEVICE_DATA[devid][-1]['时间'])
            time_diff = (current_state['时间'] - last_timestamp).total_seconds()
            if time_diff < 240:
                print(f"\n- [忽略消息] 时间戳过近 ({time_diff:.0f}秒)，非计划内数据。")
                return
        
        print(f"\n📨 收到设备【{devid}】数据 (cmd:{cmd}) @ {current_state['时间'].strftime('%H:%M:%S')}")
        print(f"   -> 室内温度: {current_state['室内温度']}°C, 设置温度: {current_state['设置温度']}°C, 风速: {current_state['风速']}档")
        print(f"   -> 室内湿度: {current_state['室内湿度']}%, 供水温度: {current_state['供水温度']}°C, 回水温度: {current_state['回水温度']}°C")

        if devid not in DEVICE_DATA: DEVICE_DATA[devid] = []
        DEVICE_DATA[devid].append(current_state)
        if len(DEVICE_DATA[devid]) > TIME_STEPS:
            DEVICE_DATA[devid].pop(0)
        
        print(f"   -> 当前数据点数量: {len(DEVICE_DATA[devid])}/{TIME_STEPS}")

        can_predict = (time.time() - LAST_PREDICTION_TIME.get(devid, 0)) > 590

        if len(DEVICE_DATA[devid]) >= TIME_STEPS and can_predict:
            print(f"--- 数据窗口已满，触发智能预测与调节 ---")
            LAST_PREDICTION_TIME[devid] = time.time()
            
            current_temp = current_state['室内温度']
            set_temp = current_state['设置温度']
            current_fan_speed = current_state['风速']
            
            initialize_speed_residuals(devid)

            if devid in LAST_PRED:
                last_pred_data = LAST_PRED[devid]
                last_prediction = last_pred_data['predicted_temp']
                last_fan_speed = last_pred_data['fan_speed']
                
                update_smart_residual(
                    devid, current_temp, last_prediction, current_fan_speed, last_fan_speed
                )
                
                log_entry = {
                    "时间": current_state["时间"], 
                    "预测温度": last_prediction, 
                    "实际温度": current_temp,
                    "风速": last_fan_speed,
                    "设置温度": set_temp
                }
                PREDICTION_HISTORY.append(log_entry)
            
            print("   -> 正在加载最终版模型 lgbm_hvac_model_golden.pkl...")
            model = joblib.load(os.path.join(MODEL_DIR, 'lgbm_hvac_model_golden.pkl'))
            
            print("   -> 正在加载特征列表 features.pkl...")
            feature_names = joblib.load(os.path.join(MODEL_DIR, 'features_golden.pkl'))
            
            outdoor_temp, humidity, radiation = get_outdoor_conditions(DEVICE_LATITUDE, DEVICE_LONGITUDE)
            df_window = pd.DataFrame(DEVICE_DATA[devid])
            df_window.loc[df_window.index[-1], '室外温度'] = outdoor_temp
            df_window.loc[df_window.index[-1], '室外湿度'] = humidity
            df_window.loc[df_window.index[-1], '光照强度'] = radiation
            
            fan_speed_predictions = predict_with_different_fan_speeds(df_window, model, feature_names, current_state, ROOM_CONFIG, PHYSICS_CONFIG)
            
            optimal_fan_speed, predicted_temp = decide_optimal_fan_speed(
                current_temp, set_temp, fan_speed_predictions, devid, current_fan_speed
            )
            
            LAST_PRED[devid] = {
                'predicted_temp': predicted_temp,
                'fan_speed': optimal_fan_speed
            }
            LAST_MODEL_CHANGE[devid] = fan_speed_predictions[optimal_fan_speed]
            
            print("\n" + "="*60)
            print(" **智能风速调节系统预测结果** ")
            print(f"   -> 当前室内温度: {current_temp:.2f}°C")
            print(f"   -> 设置温度: {set_temp:.2f}°C")
            print(f"   -> 当前风速: {current_fan_speed}档")
            print(f"   -> 推荐风速: {optimal_fan_speed}档")
            used_residual = get_residual_for_prediction(devid, optimal_fan_speed)
            print(f"   -> {optimal_fan_speed}档残差修正量: {used_residual:+.4f}°C")
            print(f"   -> **最终预测的10分钟后温度: {predicted_temp:.2f}°C**")
            print("="*60)
            
            if optimal_fan_speed != current_fan_speed and current_state['开机'] == 1:
                print(f"\n🎮 [智能调节] 检测到需要调节风速：{current_fan_speed}档 -> {optimal_fan_speed}档")
                send_command_305(client, optimal_fan_speed, devid)
            else:
                print(f"\n✅ [智能调节] 当前风速已是最优选择，无需调节")

        elif len(DEVICE_DATA[devid]) >= TIME_STEPS and not can_predict:
            print("   -> 数据窗口已满，但未到预测周期，本次跳过预测。")

    except Exception as e:
        print(f"处理消息时发生严重错误: {e}")
        print(f"原始消息: {msg.payload.decode()}")


# --- 7. 增强版绘图功能 ---
def plot_and_save_comparison():
    """根据记录的历史数据，生成预测值与实际值的对比图并保存（包含风速颜色标识）"""
    print("\n--- 正在生成智能预测与实际温度对比图 ---")
    if len(PREDICTION_HISTORY) < 2:
        print("历史数据点不足，无法生成图表。")
        return

    df = pd.DataFrame(PREDICTION_HISTORY)
    
    # 智能字体选择，兼容多平台
    my_font = None
    font_paths = ['/System/Library/Fonts/PingFang.ttc', # macOS 苹方
                  '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'] # Linux 文泉驿正黑
    for path in font_paths:
        if os.path.exists(path):
            my_font = FontProperties(fname=path)
            print(f"✅ 成功加载字体: {path}")
            break
    if not my_font:
        print("⚠️  警告: 未找到指定的中文字体，图表标签可能显示不正确。")

    # 风速颜色映射
    fan_speed_colors = {1: '#90EE90', 2: '#FFD700', 3: '#FF6347'}  # 1档:浅绿, 2档:金色, 3档:橙红
    fan_speed_labels = {1: '1档 (低风速)', 2: '2档 (中风速)', 3: '3档 (高风速)'}

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 10))

    # 绘制设置温度参考线
    if '设置温度' in df.columns:
        avg_set_temp = df['设置温度'].mean()
        ax.axhline(y=avg_set_temp, color='gray', linestyle=':', linewidth=2, alpha=0.7, label=f'设置温度 ({avg_set_temp:.1f}°C)')

    # 绘制实际温度曲线
    ax.plot(df['时间'], df['实际温度'], marker='o', linestyle='-', color='royalblue', 
            linewidth=3, markersize=6, label='实际温度', zorder=3)

    # 按风速分段绘制预测温度曲线
    df['风速_shift'] = df['风速'].shift(1)  # 获取上一个点的风速，用于分段
    
    for i in range(len(df) - 1):
        current_fan_speed = df.iloc[i]['风速']
        color = fan_speed_colors.get(current_fan_speed, '#808080')
        
        ax.plot([df.iloc[i]['时间'], df.iloc[i+1]['时间']], 
                [df.iloc[i]['预测温度'], df.iloc[i+1]['预测温度']], 
                marker='x', linestyle='--', color=color, linewidth=2.5, markersize=8, zorder=2)

    # 添加风速图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='royalblue', marker='o', linestyle='-', linewidth=3, markersize=6, label='实际温度'),
        Line2D([0], [0], color='gray', linestyle=':', linewidth=2, label=f'设置温度 ({avg_set_temp:.1f}°C)')
    ]
    
    # 添加风速图例
    for speed, color in fan_speed_colors.items():
        if speed in df['风速'].values:
            legend_elements.append(
                Line2D([0], [0], color=color, marker='x', linestyle='--', linewidth=2.5, 
                       markersize=8, label=f'预测温度 - {fan_speed_labels[speed]}')
            )

    ax.set_title('智能风速调节系统 - 预测温度 vs 实际室内温度', fontproperties=my_font, fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('时间', fontproperties=my_font, fontsize=14)
    ax.set_ylabel('温度 (°C)', fontproperties=my_font, fontsize=14)
    ax.legend(handles=legend_elements, prop=my_font, fontsize=12, loc='upper right', 
              frameon=True, fancybox=True, shadow=True)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # 格式化x轴
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    fig.autofmt_xdate()

    # 添加性能统计信息
    if len(df) > 1:
        mae = np.mean(np.abs(df['预测温度'] - df['实际温度']))
        rmse = np.sqrt(np.mean((df['预测温度'] - df['实际温度'])**2))
        
        # 在图表上添加统计信息
        stats_text = f'MAE: {mae:.2f}°C\nRMSE: {rmse:.2f}°C\n数据点: {len(df)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 保存图表
    save_path = "smart_hvac_temperature_comparison.png"
    try:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 智能调节系统图表已成功保存到: {os.path.abspath(save_path)}")
    except Exception as e:
        print(f"❌ 保存图表时出错: {e}")
    
    plt.close()

# --- 8. MQTT 主循环与启动 ---
def send_command_307(client):
    print(f"\n[发送指令 307] -> 向设备 {DEVICE_ID} 请求实时数据...")
    payload = {"ver": "1.0.2", "cmd": "307", "devid": DEVICE_ID, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    client.publish(TOPIC_TO_PUBLISH, json.dumps(payload))
    print("指令 307 已发送。")

def main_loop(client):
    last_request_time = 0
    while True:
        now = time.time()
        if DEVICE_STATUS["last_seen"] != 0 and (now - DEVICE_STATUS["last_seen"] > STATUS_TIMEOUT):
            if DEVICE_STATUS["online"]:
                print(f"\n🚨 [警告] 超过 {STATUS_TIMEOUT} 秒未收到设备 {DEVICE_ID} 的数据，设备可能已离线！")
                DEVICE_STATUS["online"] = False
        if now - last_request_time > 300:
            send_command_307(client)
            last_request_time = now
        time.sleep(1)

def preload_historical_data():
    """从CSV文件加载历史数据来创建数据窗口"""
    print("--- 正在从历史数据创建数据窗口... ---")
    try:
        df_hist = pd.read_csv(REAL_DATA_PATH)
        df_hist.rename(columns={
            'timestamp': '时间', 
            'indoor_temp': '室内温度', 
            'set_temp': '设置温度', 
            'fan_speed': '风速', 
            'ac_on': '开机'
        }, inplace=True)
        required_cols = ['时间', '室内温度', '设置温度', '风速', '开机']
        df_hist = df_hist[required_cols]
        history_points = df_hist.tail(TIME_STEPS - 1).to_dict('records')
        DEVICE_DATA[DEVICE_ID] = history_points
        print(f"✅ 数据窗口已预热，包含 {len(history_points)} 条历史数据。")
    except FileNotFoundError:
        print(f"⚠️ 警告：未找到历史数据文件 '{REAL_DATA_PATH}'。")
    except Exception as e:
        print(f"⚠️ 警告：预热历史数据时出错: {e}")

if __name__ == "__main__":
    # preload_historical_data()  # 如需预热历史数据，请取消注释
    
    client = mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION2, client_id=CLIENT_ID)
    client.on_connect = on_connect
    client.on_message = on_message
    
    print(f"正在连接到服务器 {MQTT_BROKER}:{MQTT_PORT}...")
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        print(f"❌ 连接时出错: {e}")
        exit()
    
    client.loop_start()
    time.sleep(2)
    
    if not client.is_connected():
        print("无法建立连接，程序退出。")
        exit()
    
    main_thread = threading.Thread(target=main_loop, args=(client,))
    main_thread.daemon = True
    main_thread.start()
    
    print("\n" + "="*70)
    print(" 🎯 智能风速调节HVAC预测系统已启动")
    print("="*70)
    print(f"监控设备: {DEVICE_ID}")
    print(f"设备坐标: ({DEVICE_LATITUDE}, {DEVICE_LONGITUDE})")
    print("系统功能:")
    print("   每5分钟自动获取设备数据")
    print("   每10分钟进行温度预测")
    print("   分析1/2/3档风速效果")
    print("   自动调节风速以接近设定温度")
    print("   图表显示不同风速预测效果")
    print("按 CTRL+C 退出程序并生成分析图表。")
    print("="*70)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n程序正在退出...")
        plot_and_save_comparison()
        print("\n--- 最终残差性能分析 ---")
        for device_id in SPEED_BASED_EMA_RESIDUAL.keys():
            analyze_residual_performance(device_id)
        client.loop_stop()
        client.disconnect()
        print("已断开连接。程序已安全退出。")