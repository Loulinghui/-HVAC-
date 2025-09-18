import joblib
import lightgbm as lgb # 虽然不直接用，但joblib加载LGBM模型时需要它被导入

# --- 配置 ---
# 您已经训练好的模型文件名
#model_filename = "lgbm_hvac_model_final.pkl" 
model_filename = "lgbm_hvac_model_final_2.pkl" 

# 您想要保存特征列表的文件名
#features_filename = "features.pkl" 
features_filename = "features_2.pkl" 

print(f"正在从模型文件 '{model_filename}' 中提取特征列表...")

try:
    # 1. 加载已经训练好的模型
    loaded_model = joblib.load(model_filename)
    print("模型加载成功！")

    # 2. 从加载的模型中提取特征名称
    # 对于LightGBM模型，特征列表存储在 booster_ 对象的 feature_name() 方法中
    feature_list = loaded_model.booster_.feature_name()
    print("特征列表提取成功！")
    
    # 打印一些信息以供核对
    num_features = len(feature_list)
    print(f"共提取到 {num_features} 个特征。")
    print("\n特征列表 (前10个):")
    for i, feature in enumerate(feature_list[:10]):
        print(f"  {i+1}. {feature}")
    
    # 3. 将特征列表保存到新的 .pkl 文件中
    joblib.dump(feature_list, features_filename)
    print(f"\n✅ 特征列表已成功保存到文件: '{features_filename}'")

except FileNotFoundError:
    print(f"错误: 找不到模型文件 '{model_filename}'。请确保此脚本与模型文件在同一个目录下。")
except Exception as e:
    print(f"提取过程中发生错误: {e}")
    print("请确保您已安装 lightgbm 库 (pip install lightgbm)。")