# convert_lgbm_to_onnx.py
import joblib
import sys
import os
import lightgbm as lgb
import numpy as np
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxruntime as rt

def convert_to_onnx(pkl_path, onnx_path=None):
    # 1. 检查文件
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"找不到模型文件: {pkl_path}")

    # 2. 加载模型
    model = joblib.load(pkl_path)
    print(f"模型类型: {type(model)}")

    # 3. 获取特征数
    if hasattr(model, "n_features_"):
        feature_count = model.n_features_
    elif hasattr(model, "num_feature"):
        feature_count = model.num_feature()
    else:
        raise ValueError("无法识别模型特征数")

    print(f"特征数: {feature_count}")

    # 如果是 Booster，需要包成 sklearn API 模型
    if isinstance(model, lgb.Booster):
        print("检测到 Booster 模型，转换为 LGBMRegressor 包装...")
        tmp_model = lgb.LGBMRegressor()
        tmp_model._Booster = model
        model = tmp_model

    # 4. 定义 ONNX 输入类型
    initial_type = [('input', FloatTensorType([None, feature_count]))]

    # 5. 转换
    onnx_model = convert_lightgbm(model, initial_types=initial_type)

    # 6. 保存
    if onnx_path is None:
        onnx_path = os.path.splitext(pkl_path)[0] + ".onnx"

    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"✅ 转换完成: {onnx_path}")

    # 7. 验证 ONNX 推理
    print("开始验证 ONNX 模型推理...")
    sess = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # 随机生成一条测试数据
    x = np.random.rand(1, feature_count).astype(np.float32)
    y_pred = sess.run(None, {input_name: x})

    print(f"ONNX 推理输出: {y_pred}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python convert_lgbm_to_onnx.py model.pkl [model.onnx]")
    else:
        pkl_file = sys.argv[1]
        onnx_file = sys.argv[2] if len(sys.argv) > 2 else None
        convert_to_onnx(pkl_file, onnx_file)
