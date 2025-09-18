# --- 核心依赖库 ---
# 请先确保已安装以下库:
# pip install pandas numpy tqdm scikit-learn lightgbm optuna
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import warnings
from tqdm import tqdm
import joblib
import os
import optuna

# --- 机器学习库 ---
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
# Optuna的日志比较冗长，可以设置为只看警告
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# --- 第1部分: 增强型HVAC数据模拟 (保持不变) ---
# =============================================================================

# --- 1.1 全局配置模块 ---
ROOM_CONFIG = {
    '财务室': {'面积': 23.21, "最大总冷量": 1654, "最大显冷量": 1198 ,'风盘型号': 'FP-136', '制冷高系数': 5.99, '制冷中系数': 5, '制冷低系数': 3.59, '制热高系数': 8.42, '制热中系数': 6.82, '制热低系数': 4.8},
    '董事长办公室': {'面积': 40.71, "最大总冷量": 1654, "最大显冷量": 1198 ,'风盘型号': 'FP-136', '制冷高系数': 5.99, '制冷中系数': 5, '制冷低系数': 3.59, '制热高系数': 8.42, '制热中系数': 6.82, '制热低系数': 4.8},
    '南区总室': {'面积': 12.08, "最大总冷量": 624, "最大显冷量": 446 ,'风盘型号': 'FP-51', '制冷高系数': 2.23, '制冷中系数': 1.82, '制冷低系数': 1.35, '制热高系数': 3.23, '制热中系数': 2.72, '制热低系数': 2.01},
    '软件办公室': {'面积': 38.52, "最大总冷量": 1038, "最大显冷量": 742 ,'风盘型号': 'FP-85', '制冷高系数': 3.71, '制冷中系数': 3.03, '制冷低系数': 2.2, '制热高系数': 5.17, '制热中系数': 4.18, '制热低系数': 3.00},
    '设计部经理办公室': {'面积': 9.91, "最大总冷量": 624, "最大显冷量": 446 ,'风盘型号': 'FP-51', '制冷高系数': 2.23, '制冷中系数': 1.82, '制冷低系数': 1.35, '制热高系数': 3.23, '制热中系数': 2.72, '制热低系数': 2.01},
    '市场部公共办公室': {'面积': 108.43, "最大总冷量": 1190, "最大显冷量": 856 , '风盘型号': 'FP-102', '制冷高系数': 4.28, '制冷中系数': 3.58, '制冷低系数': 2.58, '制热高系数': 6.28, '制热中系数': 5.08, '制热低系数': 3.51},
    '市场部经理办公室': {'面积': 20.95, "最大总冷量": 1654, "最大显冷量": 1198 ,'风盘型号': 'FP-136', '制冷高系数': 5.99, '制冷中系数': 5, '制冷低系数': 3.59, '制热高系数': 8.42, '制热中系数': 6.82, '制热低系数': 4.8},
    '四楼接待室': {'面积': 32.29, "最大总冷量": 1190, "最大显冷量": 856 ,'风盘型号': 'FP-102', '制冷高系数': 4.28, '制冷中系数': 3.58, '制冷低系数': 2.58, '制热高系数': 6.28, '制热中系数': 5.08, '制热低系数': 3.51},
    '行政人事部': {'面积': 34.59, "最大总冷量": 1654, "最大显冷量": 1198 ,'风盘型号': 'FP-136', '制冷高系数': 5.99, '制冷中系数': 5, '制冷低系数': 3.59, '制热高系数': 8.42, '制热中系数': 6.82, '制热低系数': 4.8},
    '小会议室': {'面积': 32.57, "最大总冷量": 1038, "最大显冷量": 742 ,'风盘型号': 'FP-85', '制冷高系数': 3.71, '制冷中系数': 3.03, '制冷低系数': 2.2, '制热高系数': 5.17, '制热中系数': 4.18, '制热低系数': 3.00},
    '生产部经理室': {'面积': 32.68, "最大总冷量": 624, "最大显冷量": 446 ,'风盘型号': 'FP-51', '制冷高系数': 2.23, '制冷中系数': 1.82, '制冷低系数': 1.35, '制热高系数': 3.23, '制热中系数': 2.72, '制热低系数': 2.01},
    '仓管部': {'面积': 30.35, "最大总冷量": 1190, "最大显冷量": 856,'风盘型号': 'FP-102', '制冷高系数': 4.28, '制冷中系数': 3.58, '制冷低系数': 2.58, '制热高系数': 6.28, '制热中系数': 5.08, '制热低系数': 3.51},
    '研发部': {'面积': 45.64, "最大总冷量": 1190, "最大显冷量": 856,'风盘型号': 'FP-102', '制冷高系数': 4.28, '制冷中系数': 3.58, '制冷低系数': 2.58, '制热高系数': 6.28, '制热中系数': 5.08, '制热低系数': 3.51},
    '成品仓库': {'面积': 80.49, "最大总冷量": 1868, "最大显冷量": 1372 , '风盘型号': 'FP-170', '制冷高系数': 6.86, '制冷中系数': 6.05, '制冷低系数': 4.58, '制热高系数': 9.84, '制热中系数': 8.56, '制热低系数': 6.40},
    '测试间': {'面积': 48.48, "最大总冷量": 2248, "最大显冷量": 1732 , '风盘型号': 'FP-204', '制冷高系数': 8.66, '制冷中系数': 7.21, '制冷低系数': 4.94, '制热高系数': 11.84, '制热中系数': 9.71, '制热低系数': 6.27}
}


SIMULATION_CONFIG = {"OPERATION_MODE": "energy_storage"}
LOCATION_CONFIG = {"TIMEZONE": "Asia/Shanghai"}
COOLING_MONTHS = [5, 6, 7, 8, 9]

WATER_SYSTEM_CONFIG = {
    "ENERGY_STORAGE": {"NIGHT_CHARGE_TEMP": 10.0, "STORAGE_CAPACITY": 1000, "HEAT_GAIN_RATE": 0.02, "USAGE_PERIODS": [{"start": 8, "end": 12, "mode": "storage_only"}, {"start": 12, "end": 15.5, "mode": "mixed"}, {"start": 15.5, "end": 18, "mode": "storage_only"}]},
    "DIRECT_CHILLER": {"SUPPLY_TEMP": 7.0, "TEMP_VARIATION": 1.0}
}


PHYSICS_CONFIG = {
    "INSULATION_FACTOR": 0.008, 
    "HEAT_GAIN_EFFECT_FACTOR": 0.003, 
    "SOLAR_HEAT_GAIN_FACTOR": 0.8, 
    "EVENT_PROBABILITY": 0.05, 
    "THERMAL_MASS_FACTOR": 0.7, 
    "OUTDOOR_INFILTRATION": 0.15,
    "WATTS_TO_TEMP_CHANGE_FACTOR": 300 / (1.2 * 2.8 * 1005),
    "WATTS_TO_HUMIDITY_CHANGE_FACTOR": 0.0005, 
    "FAN_SPEED_U_X_MULTIPLIERS": {3: 1.0, 2: 0.75, 1: 0.50, 0: 0.0},
}

def load_weather_data_from_local_file(filepath, timezone):
    print(f"正在从本地文件加载天气数据: {filepath}...")
    try:
        df = pd.read_csv(filepath)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        df = df.tz_localize(timezone) if df.index.tz is None else df.tz_convert(timezone)
        df.rename(columns={'out_temp': 'temperature', 'out_humidity': 'humidity', 'out_solarradiation': 'light_intensity'}, inplace=True)
        return df[['temperature', 'humidity', 'light_intensity']]
    except FileNotFoundError:
        print(f"错误: 未找到天气数据文件 '{filepath}'。请确保文件路径正确。")
        return None

# =============================================================================
# --- 请将此 WaterSupplyManager 类完整替换掉脚本中的旧版本 ---
# =============================================================================
class WaterSupplyManager:
    """
    【最终修正版】供水管理器
    - 移除了TEMP_EFFICIENCY_MAP
    - 采用分段函数来精确模拟效率因子g(t)
    """
    def __init__(self, operation_mode, water_config):
        self.operation_mode = operation_mode
        self.water_config = water_config
        self.storage_temp = water_config["ENERGY_STORAGE"]["NIGHT_CHARGE_TEMP"]
        self.current_storage = water_config["ENERGY_STORAGE"]["STORAGE_CAPACITY"]

    def get_supply_temperature(self, current_time, outdoor_temp, total_cooling_load):
        # 这个函数保持不变
        hour = current_time.hour + current_time.minute / 60.0
        if self.operation_mode == "energy_storage":
            config = self.water_config["ENERGY_STORAGE"]
            if hour >= 23 or hour < 7:
                self.storage_temp = config["NIGHT_CHARGE_TEMP"]
                self.current_storage = config["STORAGE_CAPACITY"]
                return self.storage_temp
            current_mode = "off"
            for period in config["USAGE_PERIODS"]:
                if period["start"] <= hour < period["end"]:
                    current_mode = period["mode"]
                    break
            if current_mode == "storage_only":
                self.current_storage -= min(total_cooling_load / 100, self.current_storage / 100)
                temp_rise = (1 - self.current_storage / config["STORAGE_CAPACITY"]) * 3.0
                self.storage_temp = config["NIGHT_CHARGE_TEMP"] + temp_rise
                return self.storage_temp
            elif current_mode == "mixed":
                self.current_storage -= min(total_cooling_load * 0.5 / 100, self.current_storage / 100)
                storage_temp = config["NIGHT_CHARGE_TEMP"] + (1 - self.current_storage / config["STORAGE_CAPACITY"]) * 2.0
                self.storage_temp = storage_temp * 0.5 + 12.0 * 0.5
                return self.storage_temp
        else: # direct_chiller
            base_temp = self.water_config["DIRECT_CHILLER"]["SUPPLY_TEMP"]
            variation = self.water_config["DIRECT_CHILLER"]["TEMP_VARIATION"]
            return base_temp + np.random.uniform(-variation / 2, variation / 2)
        return self.storage_temp

    def get_cooling_efficiency_factor(self, supply_temp):
        """
        【核心修正】使用分段函数来模拟效率：
        - 14°C以下：按专家公式 7/i 计算
        - 14°C以上：效率线性、快速地衰减至0
        """
        RATED_SUPPLY_TEMP = 7.0
        
        # 安全检查
        if supply_temp <= 0:
            return 0

        # 1. 理想工作区 (14°C以下)
        if supply_temp <= 14.0:
            efficiency_factor = RATED_SUPPLY_TEMP / supply_temp
            # 效率上限设为120%，防止温度过低时效率无限增大
            return min(efficiency_factor, 1.2)
        
        # 2. 效能骤降区 (14°C以上)
        else:
            # 在这个区间，我们设计一个线性衰减函数
            # 它从14°C时的效率值(7/14=0.5)开始，到22°C时衰减到0
            # y = mx + c.  点1:(14, 0.5), 点2:(22, 0)
            # m = (0 - 0.5) / (22 - 14) = -0.0625
            # c = 0.5 - (-0.0625 * 14) = 1.375
            efficiency_factor = -0.0625 * supply_temp + 1.375
            
            # 确保效率不会低于0，这样就永远不会制热 
            return max(0, efficiency_factor)


# =============================================================================
# --- 1.3 核心模拟器 (最终黄金版) ---
# =============================================================================
class EnhancedHVACSimulator:
    def __init__(self, room_config, physics_config, cooling_months, historical_weather_df, operation_mode):
        self.room_config, self.physics_config, self.cooling_months, self.historical_weather_df = room_config, physics_config, cooling_months, historical_weather_df
        self.data, self.room_states = [], {}
        self.water_manager = WaterSupplyManager(operation_mode, WATER_SYSTEM_CONFIG)
        self.anomaly_probability, self.human_intervention_probability, self.physics_lesson_probability = 0.01, 0.02, 0.002
        self.active_anomalies, self.active_interventions, self.active_physics_lesson = {}, {}, {}
        print(f"仿真器初始化完成，运行模式: {operation_mode} (已集成专家公式与所有高级模拟功能)")
    def _get_outdoor_conditions(self, dt):
        weather_now = self.historical_weather_df.asof(pd.Timestamp(dt, tz=self.historical_weather_df.index.tz))
        return weather_now['temperature'], weather_now['humidity'], weather_now['light_intensity']
    def _introduce_system_anomalies(self, room_name, current_time):
        if room_name in self.active_anomalies:
            if current_time < self.active_anomalies[room_name]['end_time']: return self.active_anomalies[room_name]['type']
            else: del self.active_anomalies[room_name]
        if np.random.random() < self.anomaly_probability:
            anomaly_type = np.random.choice(['chiller_inefficiency', 'low_flow'])
            self.active_anomalies[room_name] = {'type': anomaly_type, 'end_time': current_time + timedelta(hours=np.random.uniform(1.5, 4))}
            return anomaly_type
        return "normal"
    def _introduce_human_intervention(self, room_name, current_time, logical_fan_speed):
        if room_name in self.active_interventions:
            if current_time < self.active_interventions[room_name]['end_time']: return self.active_interventions[room_name]['fan_speed']
            else: del self.active_interventions[room_name]
        if np.random.random() < self.human_intervention_probability and logical_fan_speed != 0:
            overridden_fan_speed = np.random.choice([s for s in [0, 1, 2, 3] if s != logical_fan_speed])
            self.active_interventions[room_name] = {'fan_speed': overridden_fan_speed, 'end_time': current_time + timedelta(minutes=int(np.random.choice([30, 60, 90, 120])))}
            return overridden_fan_speed
        return logical_fan_speed
    def _get_hvac_action(self, state):
        current_temp, set_temp = state['current_temp'], state['target_temp']
        if current_temp <= set_temp - 0.5: return 0
        if current_temp >= set_temp + 0.5:
            temp_diff = current_temp - set_temp
            if temp_diff > 4: return 3
            elif 2 <= temp_diff <= 4: return 2
            else: return 1
        return state['fan_speed']
    def _calculate_total_cooling_load(self):
        total_load = 0
        u_x_map = self.physics_config["FAN_SPEED_U_X_MULTIPLIERS"]
        for name, state in self.room_states.items():
            if state.get('is_on', False) and state['fan_speed'] > 0:
                max_cooling = self.room_config[name]['最大总冷量']
                total_load += max_cooling * u_x_map.get(state['fan_speed'], 0)
        return total_load
    def simulate_room(self, room_name, start_time, end_time):
        o_temp, o_hum, _ = self._get_outdoor_conditions(start_time)
        self.room_states[room_name] = {'name': room_name, 'area': self.room_config[room_name]['面积'], 'current_temp': o_temp + np.random.uniform(-2, 2), 'target_temp': 24.0, 'fan_speed': 0, 'indoor_humidity': o_hum + np.random.uniform(-5, 5)}
        state = self.room_states[room_name]
        current_time = start_time
        pbar = tqdm(total=int((end_time - start_time).total_seconds() / 300), desc=f"模拟 {room_name}", leave=False)
        while current_time < end_time:
            state['timestamp'] = current_time
            is_work = 8 <= current_time.hour < 18 and current_time.weekday() < 5
            state['target_temp'] = 24.0 if is_work else 26.0
            lesson_state = self.active_physics_lesson.get(room_name)
            if lesson_state:
                o_temp, o_hum, light = lesson_state['frozen_weather']
                final_fan_speed = lesson_state['sequence'][lesson_state['step']]
                lesson_state['step'] += 1
                if lesson_state['step'] >= len(lesson_state['sequence']): del self.active_physics_lesson[room_name]
            else:
                if np.random.random() < self.physics_lesson_probability:
                    o_temp, o_hum, light = self._get_outdoor_conditions(current_time)
                    self.active_physics_lesson[room_name] = {'frozen_weather': (o_temp, o_hum, light), 'sequence': [3, 2, 1, 0], 'step': 0}
                    current_time += timedelta(minutes=5); pbar.update(1); continue
                anomaly_status = self._introduce_system_anomalies(room_name, current_time)
                o_temp, o_hum, light = self._get_outdoor_conditions(current_time)
                logical_fan_speed = self._get_hvac_action(state)
                final_fan_speed = self._introduce_human_intervention(room_name, current_time, logical_fan_speed)
            state['fan_speed'] = final_fan_speed
            is_on = final_fan_speed > 0
            heat_gains = state['area'] * (30 if is_work else 8)
            temp_change = (o_temp - state['current_temp']) * self.physics_config["INSULATION_FACTOR"] + heat_gains * self.physics_config["HEAT_GAIN_EFFECT_FACTOR"] / state['area']
            supply_temp = 18
            if is_on and current_time.month in COOLING_MONTHS:
                supply_temp = self.water_manager.get_supply_temperature(current_time, o_temp, self._calculate_total_cooling_load())
                if 'anomaly_status' in locals() and anomaly_status == 'chiller_inefficiency': supply_temp = max(supply_temp, np.random.uniform(14, 17))
                g_t = self.water_manager.get_cooling_efficiency_factor(supply_temp)
                u_x = self.physics_config["FAN_SPEED_U_X_MULTIPLIERS"].get(final_fan_speed, 0)
                room_hvac_config = self.room_config[room_name]
                max_sensible_watts, max_total_watts = room_hvac_config['最大显冷量'], room_hvac_config['最大总冷量']
                effective_sensible_watts = max_sensible_watts * g_t * u_x
                effective_latent_watts = (max_total_watts - max_sensible_watts) * g_t * u_x
                if 'anomaly_status' in locals() and anomaly_status == 'low_flow':
                    effective_sensible_watts *= 0.3; effective_latent_watts *= 0.3
                temp_change -= effective_sensible_watts * self.physics_config['WATTS_TO_TEMP_CHANGE_FACTOR'] / state['area']
                state['indoor_humidity'] -= effective_latent_watts * self.physics_config['WATTS_TO_HUMIDITY_CHANGE_FACTOR']
            state['indoor_humidity'] += (o_hum - state['indoor_humidity']) * 0.05
            state['indoor_humidity'] = np.clip(state['indoor_humidity'], 35, 95)
            record = {'时间': current_time, '名称': room_name, '室内温度': round(state['current_temp'], 1), '设置温度': round(state['target_temp'], 1), '风速': final_fan_speed, '开机': int(is_on), '室外温度': round(o_temp, 1), '室外湿度': round(o_hum, 1), '光照强度': round(light, 0), '供水温度': round(supply_temp, 1), '室内湿度': round(state['indoor_humidity'], 1), '未来10分钟室内温度': round(state['current_temp'] + temp_change * 2, 1)}
            self.data.append(record)
            state['current_temp'] += temp_change + np.random.uniform(-0.1, 0.1)
            state['current_temp'] = np.clip(state['current_temp'], 15, 40)
            current_time += timedelta(minutes=5)
            pbar.update(1)
        pbar.close()
    def generate_dataset(self, start_date, end_date):
        for room_name in tqdm(self.room_config.keys(), desc="总体进度"): self.simulate_room(room_name, start_date, end_date)
        columns = ['时间', '名称', '室内温度', '设置温度', '风速', '开机', '室外温度', '室外湿度', '室内湿度', '光照强度', '供水温度', '未来10分钟室内温度']
        df = pd.DataFrame(self.data, columns=columns).sort_values(['名称', '时间']).reset_index(drop=True)
        return df
# =============================================================================
# --- 第2部分: 机器学习模型 (最终修正版，采用双重约束) ---
# =============================================================================

# --- 2. 最终版特征工程与模型训练 ---


# --- 2. 特征工程 (黄金版) ---
def feature_engineering_advanced(df, room_config, physics_config):
    print("正在进行高级特征工程...")
    df = df.copy()
    df['时间'] = pd.to_datetime(df['时间'])
    df['小时'], df['星期几'], df['月'] = df['时间'].dt.hour, df['时间'].dt.dayofweek, df['时间'].dt.month
    df['温差'], df['室内外温差'] = df['室内温度'] - df['设置温度'], df['室内温度'] - df['室外温度']
    df['室内外湿度差'] = df['室内湿度'] - df['室外湿度']
    df['设备负荷'] = df['开机'] * df['风速']
    df['是否工作时间'] = ((df['小时'] >= 8) & (df['小时'] < 18)).astype(int)
    water_manager = WaterSupplyManager(None, WATER_SYSTEM_CONFIG)
    df['g_t_efficiency'] = df['供水温度'].apply(water_manager.get_cooling_efficiency_factor)
    u_x_map = physics_config["FAN_SPEED_U_X_MULTIPLIERS"]
    df['u_x_efficiency'] = df['风速'].map(u_x_map).fillna(0)
    def get_max_sensible_cooling(room_name):
        return room_config.get(str(room_name), {}).get('最大显冷量', 0)
    df['max_sensible_cooling'] = df['名称'].apply(get_max_sensible_cooling)
    df['有效制冷潜力'] = df['max_sensible_cooling'] * df['g_t_efficiency'] * df['u_x_efficiency']
    for lag in [1, 2, 3]: df[f'室内温度_lag_{lag}'] = df.groupby('名称')['室内温度'].shift(lag)
    for window in [3, 6]:
        df[f'室内温度_ma_{window}'] = df.groupby('名称')['室内温度'].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'温差_ma_{window}'] = df.groupby('名称')['温差'].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    df = df.drop(columns=['g_t_efficiency', 'u_x_efficiency', 'max_sensible_cooling'])
    return df

def calibrate_and_get_target_estimators(X_train_sample, y_train_sample, target_size_kb=500, adjustment_factor=0.45):
    print("\n--- 校准模型大小以确定 n_estimators 目标范围 ---")
    adjusted_target_kb = target_size_kb * adjustment_factor
    print(f"原始目标: {target_size_kb} KB, 修正后校准目标: {adjusted_target_kb:.2f} KB。")
    base_estimators = 100
    calibration_model = lgb.LGBMRegressor(n_estimators=base_estimators, num_leaves=40, random_state=42)
    calibration_model.fit(X_train_sample, y_train_sample)
    temp_model_file = "temp_calibration_model.pkl"
    joblib.dump(calibration_model, temp_model_file)
    size_kb = os.path.getsize(temp_model_file) / 1024
    os.remove(temp_model_file)
    kb_per_estimator = size_kb / base_estimators
    target_n_estimators = int(adjusted_target_kb / kb_per_estimator)
    print(f"校准结果: 每棵树约 {kb_per_estimator:.2f} KB, 推荐 n_estimators 约: {target_n_estimators}")
    return target_n_estimators

def tune_lgbm_hyperparameters(X_train, y_train, X_val, y_val, target_n_estimators):
    n_estimators_min = max(100, target_n_estimators - 75)
    n_estimators_max = target_n_estimators + 75
    print(f"Optuna 将在 [{n_estimators_min}, {n_estimators_max}] 范围内搜索 'n_estimators'...")
    def objective(trial):
        params = {
            'objective': 'regression_l1', 'metric': 'mae', 'random_state': 42, 'n_jobs': -1,
            'n_estimators': trial.suggest_int('n_estimators', n_estimators_min, n_estimators_max),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 45),
            'max_depth': trial.suggest_int('max_depth', 5, 12),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='mae', callbacks=[lgb.early_stopping(15, verbose=False)])
        return mean_absolute_error(y_val, model.predict(X_val))
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    print(f"\n调优完成！最佳MAE (验证集): {study.best_value:.4f}")
    print(f"找到的最佳超参数: {study.best_params}")
    return study.best_params

def train_and_evaluate_model(df):
    print("\n" + "="*50 + "\n--- 开始模型训练、评估与保存流程 (黄金版) ---\n" + "="*50)
    df['温度变化量'] = df['未来10分钟室内温度'] - df['室内温度']
    df = df.dropna(subset=['未来10分钟室内温度'])
    df_featured = feature_engineering_advanced(df, ROOM_CONFIG, PHYSICS_CONFIG)
    features = [col for col in df_featured.columns if col not in ['时间', '名称', '未来10分钟室内温度', '温度变化量']]
    target = '温度变化量'
    df_final = df_featured[features + [target]].dropna()
    X, y = df_final[features], df_final[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42, shuffle=False)
    print(f"数据拆分完成: 训练集({X_train.shape[0]}), 验证集({X_val.shape[0]}), 测试集({X_test.shape[0]})")

    target_estimators = calibrate_and_get_target_estimators(X_train, y_train, target_size_kb=500, adjustment_factor=0.45)
    best_params = tune_lgbm_hyperparameters(X_train, y_train, X_val, y_val, target_estimators)
    
    print("\n--- 使用最优参数训练最终模型 ---")
    X_train_full, y_train_full = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
    final_lgbm = lgb.LGBMRegressor(**best_params, random_state=42, objective='regression_l1', metric='mae')
    final_lgbm.fit(X_train_full, y_train_full)
    
    print("\n--- 在测试集上评估最终模型性能 ---")
    preds = final_lgbm.predict(X_test)
    mae, r2 = mean_absolute_error(y_test, preds), r2_score(y_test, preds)
    print(f"  评估结果: MAE={mae:.4f}, R²={r2:.4f}")

    print("\n--- 保存模型并检查大小 ---")
    model_filename = "lgbm_hvac_model_golden.pkl"
    feature_filename = "features_golden.pkl"
    joblib.dump(final_lgbm, model_filename)
    joblib.dump(features, feature_filename)
    print(f"模型已保存到 {model_filename}, 特征列表已保存到 {feature_filename}")
    model_size_kb = os.path.getsize(model_filename) / 1024
    print(f"最终模型文件大小: {model_size_kb:.2f} KB")



# =============================================================================
# --- 主程序入口 ---
# =============================================================================
if __name__ == "__main__":
    # --- 第1步: 确定关键文件名和年份 ---
    local_weather_file = 'Nanyang_merged_sorted_cleaned_光照强度.csv'
    
    # 【修正】逻辑调整：首先加载天气数据，以确定唯一的真实年份。这是所有后续操作的基础。
    print("--- 步骤 1a: 加载天气数据以确定模拟年份 ---")
    historical_df = load_weather_data_from_local_file(filepath=local_weather_file, timezone=LOCATION_CONFIG['TIMEZONE'])
    
    if historical_df is None:
        print("致命错误: 无法加载天气数据，程序终止。")
        exit() # 如果天气数据不存在，则无法继续

    data_year = historical_df.index[0].year
    print(f"从天气数据中检测到年份为: {data_year}。")

    # 【修正】逻辑调整：现在使用这个唯一的真实年份来定义日期和最终的数据集文件名。
    start_dt = datetime(data_year, 1, 1)
    end_dt = datetime(data_year, 7, 15)
    dataset_filename = f"hvac_simulated_data_{start_dt.date()}_to_{end_dt.date()}_3.csv"
    print(f"将要检查或创建的数据集文件: {dataset_filename}")

    # --- 步骤 1b: 检查或生成数据集 ---
    
    # 【修正】现在这个检查逻辑可以正确工作了，因为它使用的文件名是基于真实年份构造的。
    if os.path.exists(dataset_filename):
        print(f"\n检测到已存在数据集文件，正在直接加载...")
        final_dataset = pd.read_csv(dataset_filename)
        # 确保时间列是datetime类型，以防万一
        final_dataset['时间'] = pd.to_datetime(final_dataset['时间'])
        print("数据集加载完成！")
    else:
        print("\n未找到现有数据集，开始生成新数据集...")
        
        simulator = EnhancedHVACSimulator(
            ROOM_CONFIG, 
            PHYSICS_CONFIG, 
            COOLING_MONTHS, 
            historical_df, # 直接使用已经加载好的天气数据
            SIMULATION_CONFIG["OPERATION_MODE"]
        )
        final_dataset = simulator.generate_dataset(start_dt, end_dt)
        
        # 使用正确的文件名保存新生成的数据集
        final_dataset.to_csv(dataset_filename, index=False, encoding='utf-8-sig')
        print(f"\n新生成的数据集已保存为: {dataset_filename}")

    print("\n--- 数据样本 (前5条) ---")
    print(final_dataset.head())
    print(f"\n数据集总行数: {len(final_dataset)}")

    # --- 第2步: 在生成的数据上进行模型训练、调优和评估 ---
    if not final_dataset.empty:
        train_and_evaluate_model(final_dataset)
    else:
        print("数据集为空，跳过模型训练。")