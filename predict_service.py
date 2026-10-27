# ============================================================================
# PySpark 独立预测服务 - 解决 Windows + Flask + Spark 兼容性问题
# ============================================================================
# 
# 功能：在独立进程中运行 Spark 预测服务，避免 Web 应用中的 Worker 连接问题
# 使用方法：
#   1. 启动预测服务：python predict_service.py
#   2. 服务地址：http://localhost:5001
#   3. 健康检查：GET http://localhost:5001/health
#   4. 预测接口：POST http://localhost:5001/predict
#
# ============================================================================

import os
import sys
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

# ============================================================================
# 【Windows Spark 环境修复】必须在 import pyspark 之前执行
# ============================================================================

python_exe = sys.executable
python_dir = os.path.dirname(python_exe)

print(f"\n🐍 Python 路径: {python_exe}")
print(f"📁 Python 目录: {python_dir}")

# 创建 python3.bat（Spark 需要这个文件）
python3_bat = os.path.join(python_dir, 'python3.bat')
if not os.path.exists(python3_bat):
    with open(python3_bat, 'w', encoding='utf-8') as f:
        f.write('@echo off\r\n')
        f.write(f'"{python_exe}" %*\r\n')
    print(f"✅ 创建 python3.bat")
else:
    print(f"️ python3.bat 已存在")

# 设置关键环境变量
os.environ['PYSPARK_PYTHON'] = 'python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'python3'
os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
os.environ['SPARK_DRIVER_HOST'] = '127.0.0.1'
os.environ['SPARK_DRIVER_BIND_ADDRESS'] = '127.0.0.1'

# 将 Python 目录添加到 PATH
if python_dir not in os.environ.get('PATH', ''):
    os.environ['PATH'] = python_dir + os.pathsep + os.environ.get('PATH', '')

print("✅ 环境变量配置完成")
print("=" * 70)

# ============================================================================
# 导入 PySpark
# ============================================================================

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

# ============================================================================
# Flask 应用
# ============================================================================

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# ============================================================================
# 全局变量
# ============================================================================

spark = None
pipeline = None
model_loaded = False

# 特征定义（与训练代码完全一致）
NUMERICAL_FEATURES = [
    'Customer_Age', 'Unit_Price', 'Quantity', 'Discount_Percent',
    'Shipping_Cost', 'Tax_Amount', 'Order_Amount', 'Delivery_Days',
    'Review_Rating', 'Customer_Lifetime_Value',
    'Profit_Margin_Percent', 'Profit_Amount'
]
CATEGORICAL_FEATURES = [
    'Customer_Gender', 'Country', 'Customer_Segment',
    'Product_Category', 'Payment_Method', 'Device_Type',
    'Traffic_Source', 'Membership_Status', 'Shipping_Method',
    'Season', 'Holiday_Season'
]

# ============================================================================
# 初始化函数
# ============================================================================

def init_spark_and_pipeline():
    """初始化 Spark 并加载 Pipeline"""
    global spark, pipeline, model_loaded
    
    print("\n" + "="*70)
    print("🔧 初始化预测服务...")
    print("="*70)
    
    try:
        # 初始化 Spark Session
        print("\n📡 正在初始化 Spark Session...")
        spark = SparkSession.builder \
            .appName("PredictService") \
            .master("local[1]") \
            .config("spark.driver.host", "127.0.0.1") \
            .config("spark.driver.bindAddress", "127.0.0.1") \
            .config("spark.driver.port", "0") \
            .config("spark.blockManager.port", "0") \
            .config("spark.shuffle.service.enabled", "false") \
            .config("spark.dynamicAllocation.enabled", "false") \
            .config("spark.network.timeout", "600s") \
            .config("spark.executor.heartbeatInterval", "60s") \
            .config("spark.python.worker.reuse", "true") \
            .config("spark.python.worker.memory", "1g") \
            .config("spark.ui.enabled", "false") \
            .getOrCreate()
        
        # 设置日志级别
        spark.sparkContext.setLogLevel("ERROR")
        print("✅ Spark 初始化成功")
        
        # 加载 Pipeline
        print("\n📦 正在加载 Pipeline...")
        pipeline_path = "return_prediction_pipeline"
        
        if os.path.exists(pipeline_path):
            pipeline = PipelineModel.load(pipeline_path)
            print(f"✅ Pipeline 加载成功")
            print(f"   Pipeline 包含 {len(pipeline.stages)} 个阶段")
            
            # 打印阶段信息
            for i, stage in enumerate(pipeline.stages[:5]):  # 只显示前5个阶段
                print(f"   [{i}] {type(stage).__name__}")
            if len(pipeline.stages) > 5:
                print(f"   ... 还有 {len(pipeline.stages) - 5} 个阶段")
            
            model_loaded = True
        else:
            print(f"❌ Pipeline 不存在: {pipeline_path}")
            print(f"   请确保已运行 ecommerce_complete_project.py 训练模型")
            model_loaded = False
            
        return model_loaded
        
    except Exception as e:
        print(f"\n 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        model_loaded = False
        return False

# ============================================================================
# 预测函数
# ============================================================================

def predict_with_pipeline(order_data):
    """使用 Pipeline 进行预测"""
    global spark, pipeline
    
    if not model_loaded or pipeline is None or spark is None:
        print("❌ 模型未加载或 Spark 未初始化")
        return None
    
    try:
        # =============================================================
        # 构建默认值
        # =============================================================
        defaults = {
            'customer_age': 42, 'unit_price': 200.0, 'quantity': 2,
            'discount_percent': 15.0, 'shipping_cost': 15.0, 'tax_amount': 12.0,
            'order_amount': 180.0, 'delivery_days': 6, 'review_rating': 3.8,
            'customer_lifetime_value': 500.0, 'profit_margin_percent': 25.0,
            'profit_amount': 45.0,
            'customer_gender': 'Female', 'country': 'United States',
            'customer_segment': 'Regular', 'product_category': 'Fashion',
            'payment_method': 'Credit Card', 'device_type': 'Mobile',
            'traffic_source': 'Website', 'membership_status': 'Silver',
            'shipping_method': 'Standard', 'season': 'Winter', 'holiday_season': 'No'
        }
        
        # 用前端数据覆盖默认值
        for key, value in order_data.items():
            if value is not None and value != '':
                defaults[key] = value
        
        # =============================================================
        # 转换字段名：snake_case -> PascalCase
        # =============================================================
        converted = {}
        for key, value in defaults.items():
            # 将 snake_case 转换为 PascalCase
            pascal_key = '_'.join(word.capitalize() for word in key.split('_'))
            
            if pascal_key in NUMERICAL_FEATURES:
                # 数值型字段强制转换为 float
                try:
                    converted[pascal_key] = float(value)
                except (ValueError, TypeError):
                    converted[pascal_key] = 0.0
            else:
                # 分类型字段转换为字符串
                converted[pascal_key] = str(value) if value is not None else ''
        
        # =============================================================
        # 创建 DataFrame
        # =============================================================
        schema_fields = []
        for col in NUMERICAL_FEATURES:
            schema_fields.append(StructField(col, DoubleType(), True))
        for col in CATEGORICAL_FEATURES:
            schema_fields.append(StructField(col, StringType(), True))
        schema = StructType(schema_fields)
        
        df = spark.createDataFrame([converted], schema=schema)
        print(f"✅ DataFrame 创建成功，包含 {len(converted)} 个特征")
        
        # =============================================================
        # 使用 Pipeline 预测
        # =============================================================
        result = pipeline.transform(df)
        pred_row = result.select('prediction', 'probability').first()
        
        if pred_row is None:
            print("❌ 预测结果为空")
            return None
        
        prediction = int(pred_row['prediction'])
        probability = float(pred_row['probability'][1])
        
        print(f"✅ 预测成功")
        print(f"   Prediction: {prediction}")
        print(f"   Probability: {probability:.4f}")
        
        return {
            'prediction': prediction,
            'probability': probability
        }
        
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# API 路由
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """健康检查接口"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model_loaded,
        'spark_ready': spark is not None,
        'service': 'PySpark Predict Service'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    try:
        data = request.get_json()
        
        if not model_loaded:
            return jsonify({
                'success': False,
                'error': '模型未加载，请检查 Pipeline 是否存在'
            }), 503
        
        result = predict_with_pipeline(data)
        
        if result is None:
            return jsonify({
                'success': False,
                'error': '预测失败，请查看服务端日志'
            }), 500
        
        # 计算风险等级
        probability = result['probability']
        risk_level = '高风险' if probability > 0.5 else ('中风险' if probability > 0.3 else '低风险')
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'probability': round(probability, 4),
            'risk_level': risk_level,
            'message': '该订单有较高退货风险' if result['prediction'] == 1 else '该订单退货风险较低',
            'model_type': 'random_forest_pipeline'
        })
        
    except Exception as e:
        print(f"❌ API 调用失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================================
# 启动服务
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 启动 PySpark 预测服务")
    print("="*70)
    
    # 初始化 Spark 和 Pipeline
    success = init_spark_and_pipeline()
    
    if success:
        print("\n🎉 模型加载成功，服务就绪！")
    else:
        print("\n⚠️ 模型加载失败，服务将以降级模式运行")
    
    print(f"\n🌐 预测服务地址: http://localhost:5001")
    print(f"🏥 健康检查: http://localhost:5001/health")
    print(f"📊 预测接口: POST http://localhost:5001/predict")
    print("="*70)
    
    # 启动 Flask 服务
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=False,
        threaded=False  # 单线程，避免 Spark 并发问题
    )
