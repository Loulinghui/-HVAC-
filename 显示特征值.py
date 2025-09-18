import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont, FontProperties

# --- 0. 环境设置 ---
# Mac 专用字体设置
try:
    # 查找系统可用的中文字体（如苹方、华文黑体等）
    mac_fonts = ['PingFang TC', 'Heiti TC', 'Songti SC', 'Arial Unicode MS']
    
    # 检查哪些字体在系统中可用
    available_fonts = []
    for font in mac_fonts:
        try:
            if findfont(FontProperties(family=font)):
                available_fonts.append(font)
        except:
            continue
    
    if available_fonts:
        # 使用第一个可用的中文字体
        plt.rcParams['font.sans-serif'] = available_fonts
        plt.rcParams['axes.unicode_minus'] = False
        print(f"✅ 已设置中文字体为: {available_fonts[0]}")
    else:
        print("⚠️ 警告: 未找到系统中安装的中文字体，将使用默认字体")
except Exception as e:
    print(f"字体设置错误: {e}")

# --- 其余代码保持不变 ---
MODEL_DIR = "" # 确保这是你存放模型的正确文件夹
# MODEL_FILENAME = 'lgbm_hvac_model_final.pkl'
# FEATURES_FILENAME = 'features.pkl'
# MODEL_FILENAME = 'lgbm_hvac_model_final_2.pkl'
# FEATURES_FILENAME = 'features_2.pkl'
# MODEL_FILENAME = 'lgbm_hvac_model_golden.pkl'
# FEATURES_FILENAME = 'features_golden.pkl'

MODEL_FILENAME = 'lgbm_hvac_model_enhanced.pkl'
FEATURES_FILENAME = 'features_enhanced.pkl'


MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
FEATURES_PATH = os.path.join(MODEL_DIR, FEATURES_FILENAME)
OUTPUT_IMAGE_PATH = 'feature_importance_full_chart.png'

def analyze_all_feature_importances():
    """
    加载已训练的模型和特征列表，分析并打印所有特征的重要性。
    """
    try:
        # --- 加载必要文件 ---
        print(f"--- 正在加载模型 '{MODEL_FILENAME}' 和特征文件 ---")
        if not all(os.path.exists(p) for p in [MODEL_PATH, FEATURES_PATH]):
            print("\n错误: 找不到模型或特征文件。")
            print(f"请确保 '{MODEL_PATH}' 和 '{FEATURES_PATH}' 文件都存在。")
            return

        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURES_PATH)
        print("✅ 文件加载成功。")

        # --- 提取并排序特征重要性 ---
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_

            # 创建一个DataFrame来展示所有特征
            feature_importance_df = pd.DataFrame({
                '特征名称': feature_names,
                '重要性分数': importances
            }).sort_values(by='重要性分数', ascending=False).reset_index(drop=True)

            # --- 关键修改：打印所有特征的排名和分数 ---
            print("\n" + "="*50)
            print("模型所有特征的重要性排名和分数")
            print("-" * 50)

            # 设置pandas以显示所有行，确保排名不会被截断
            with pd.option_context('display.max_rows', None):
                print(feature_importance_df)

            print("="*50)

            print("\n--- 正在生成特征重要性可视化图表 ---")

            df_to_plot = feature_importance_df.head(30)

            plt.figure(figsize=(18, 21))
            plt.barh(df_to_plot['特征名称'], df_to_plot['重要性分数'], color='steelblue')
            plt.gca().invert_yaxis()  # 最重要的特征在上面

            # 设置字体大小
            plt.xlabel('重要性分数', fontsize=20)
            plt.ylabel('特征名称', fontsize=20)
            plt.title('模型最重要的30个特征', fontsize=22)

            # 关键：刻度字体大小
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)

            plt.grid(axis='x', linestyle='--', alpha=0.6)
            plt.tight_layout()

            # 保存图表
            plt.savefig(OUTPUT_IMAGE_PATH, dpi=300)  # dpi=300 提高清晰度
            print(f"✅ 可视化图表已成功保存到 '{OUTPUT_IMAGE_PATH}'")

        else:
            print(f"错误: 加载的模型 '{MODEL_FILENAME}' 没有 'feature_importances_' 属性。")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")

# --- 主程序运行 ---
if __name__ == '__main__':
    analyze_all_feature_importances()
