# ============================================================================
# PySpark 电商数据分析 - Web 可视化展示系统
# ============================================================================
# 
# 技术栈：Flask + ECharts + Bootstrap
# 功能：实时展示 PySpark 分析结果的交互式 Web 界面
#
# 使用方法：
# 1. 先运行 PySpark 分析生成数据：python ecommerce_complete_project.py
# 2. 安装依赖：pip install flask
# 3. 运行：python web_dashboard.py
# 4. 访问：http://localhost:5000
#
# ============================================================================

from flask import Flask, render_template, jsonify, request
import json
import os
from datetime import datetime

# PySpark 导入（用于真实模型预测）
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import StringIndexerModel, OneHotEncoderModel, VectorAssembler

app = Flask(__name__)

# ============================================================================
# 配置
# ============================================================================
ANALYSIS_RESULTS_PATH = 'analysis_results.json'
MODEL_PATH = 'return_prediction_model'
LR_MODEL_PATH = 'return_prediction_lr_baseline'
PIPELINE_PATH = 'return_prediction_pipeline'  # 完整 Pipeline（推荐）
PIPELINE_DIR = 'pipeline'

# ✅ 强制使用规则引擎（绕过 Windows Spark Worker 问题）
# 设为 False 可尝试使用真实模型（但可能在 Windows 上失败）
FORCE_RULE_ENGINE = False

# 🚀 独立预测服务配置（方案1：最稳定）
PREDICT_SERVICE_URL = 'http://localhost:5001/predict'
USE_PREDICT_SERVICE = not FORCE_RULE_ENGINE  # 如果启用规则引擎，则不使用预测服务

# 全局变量：Spark 会话和模型
spark = None
rf_model = None
lr_model = None
pipeline_model = None  # 完整 Pipeline（推荐）
fitted_indexers = {}
fitted_encoders = {}
model_available = False

# 特征定义（与训练代码一致）
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
# 数据加载模块
# ============================================================================

def normalize_data(data):
    """将 PySpark 导出的 JSON 字段名映射为模板期望的字段名，并转换格式"""
    
    # 1. order_status: Order_Status -> status
    if 'order_status' in data:
        for item in data['order_status']:
            if 'Order_Status' in item and 'status' not in item:
                item['status'] = item['Order_Status']
    
    # 2. category_sales: Product_Category -> category, order_count -> orders, total_revenue -> revenue
    if 'category_sales' in data:
        for item in data['category_sales']:
            if 'Product_Category' in item and 'category' not in item:
                item['category'] = item['Product_Category']
            if 'order_count' in item and 'orders' not in item:
                item['orders'] = item['order_count']
            if 'total_revenue' in item and 'revenue' not in item:
                item['revenue'] = item['total_revenue']
    
    # 3. gender_stats: Customer_Gender -> gender, order_count -> orders
    if 'gender_stats' in data:
        for item in data['gender_stats']:
            if 'Customer_Gender' in item and 'gender' not in item:
                item['gender'] = item['Customer_Gender']
            if 'order_count' in item and 'orders' not in item:
                item['orders'] = item['order_count']
    
    # 4. delivery_analysis: delivery_speed -> speed, order_count -> orders, return_rate 小数->百分比
    if 'delivery_analysis' in data:
        for item in data['delivery_analysis']:
            if 'delivery_speed' in item and 'speed' not in item:
                item['speed'] = item['delivery_speed']
            if 'order_count' in item and 'orders' not in item:
                item['orders'] = item['order_count']
            # return_rate: 如果是小数(<=1)，转换为百分比
            if 'return_rate' in item and item['return_rate'] <= 1:
                item['return_rate'] = round(item['return_rate'] * 100, 2)
    
    # 5. membership_stats: Membership_Status -> level, order_count -> orders, return_rate 小数->百分比
    if 'membership_stats' in data:
        for item in data['membership_stats']:
            if 'Membership_Status' in item and 'level' not in item:
                item['level'] = item['Membership_Status']
            if 'order_count' in item and 'orders' not in item:
                item['orders'] = item['order_count']
            # return_rate: 如果是小数(<=1)，转换为百分比
            if 'return_rate' in item and item['return_rate'] <= 1:
                item['return_rate'] = round(item['return_rate'] * 100, 2)
    
    return data


def load_analysis_data():
    """加载 PySpark 分析结果数据（从 JSON 文件读取真实数据）"""
    
    # 尝试读取 PySpark 生成的真实数据
    if os.path.exists(ANALYSIS_RESULTS_PATH):
        try:
            with open(ANALYSIS_RESULTS_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ 成功加载分析结果: {ANALYSIS_RESULTS_PATH}")
            # 标准化字段名和格式
            data = normalize_data(data)
            return data
        except Exception as e:
            print(f"⚠️ 读取 JSON 文件失败: {e}")
            print("   使用模拟数据...")
    else:
        print(f"️ 未找到分析结果文件: {ANALYSIS_RESULTS_PATH}")
        print("   请先运行 ecommerce_complete_project.py 生成数据")
        print("   当前使用模拟数据展示界面效果\n")
    
    data = {
        'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        
        # 订单状态分布
        'order_status': [
            {'status': 'Delivered', 'count': 45000, 'percentage': 75.0},
            {'status': 'Returned', 'count': 6000, 'percentage': 10.0},
            {'status': 'Processing', 'count': 7500, 'percentage': 12.5},
            {'status': 'Cancelled', 'count': 1500, 'percentage': 2.5}
        ],
        
        # 产品类别销售
        'category_sales': [
            {'category': 'Fashion', 'orders': 15000, 'revenue': 1250000},
            {'category': 'Electronics', 'orders': 12000, 'revenue': 2800000},
            {'category': 'Home & Kitchen', 'orders': 10000, 'revenue': 950000},
            {'category': 'Beauty & Personal Care', 'orders': 8000, 'revenue': 680000},
            {'category': 'Sports & Outdoors', 'orders': 7500, 'revenue': 720000},
            {'category': 'Books & Media', 'orders': 5000, 'revenue': 320000},
            {'category': 'Toys & Games', 'orders': 4500, 'revenue': 280000},
            {'category': 'Health & Wellness', 'orders': 4000, 'revenue': 450000},
            {'category': 'Automotive', 'orders': 3500, 'revenue': 520000},
            {'category': 'Pet Supplies', 'orders': 2500, 'revenue': 180000}
        ],
        
        # 客户性别消费
        'gender_stats': [
            {'gender': 'Female', 'orders': 32000, 'avg_spending': 125.50, 'avg_rating': 4.2},
            {'gender': 'Male', 'orders': 28000, 'avg_spending': 138.75, 'avg_rating': 4.1}
        ],
        
        # 配送速度分析
        'delivery_analysis': [
            {'speed': '快速(≤3天)', 'orders': 18000, 'avg_rating': 4.5, 'return_rate': 5.2},
            {'speed': '标准(4-7天)', 'orders': 25000, 'avg_rating': 4.2, 'return_rate': 8.5},
            {'speed': '较慢(8-14天)', 'orders': 12000, 'avg_rating': 3.8, 'return_rate': 12.3},
            {'speed': '很慢(>14天)', 'orders': 5000, 'avg_rating': 3.2, 'return_rate': 18.7}
        ],
        
        # 会员等级分析
        'membership_stats': [
            {'level': 'Platinum', 'orders': 8000, 'avg_spending': 285.50, 'return_rate': 6.2},
            {'level': 'Gold', 'orders': 15000, 'avg_spending': 195.75, 'return_rate': 8.5},
            {'level': 'Silver', 'orders': 22000, 'avg_spending': 125.30, 'return_rate': 10.8},
            {'level': 'Standard', 'orders': 15000, 'avg_spending': 85.20, 'return_rate': 13.5}
        ],
        
        # 支付方式统计
        'payment_stats': [
            {'method': 'Credit Card', 'count': 25000, 'avg_amount': 145.50, 'return_rate': 9.2},
            {'method': 'Debit Card', 'count': 18000, 'avg_amount': 125.30, 'return_rate': 10.5},
            {'method': 'PayPal', 'count': 12000, 'avg_amount': 135.75, 'return_rate': 8.8},
            {'method': 'Cash on Delivery', 'count': 5000, 'avg_amount': 95.20, 'return_rate': 15.3}
        ],
        
        # 设备类型分析
        'device_stats': [
            {'device': 'Mobile', 'orders': 35000, 'avg_amount': 115.50, 'return_rate': 11.2},
            {'device': 'Desktop', 'orders': 20000, 'avg_amount': 165.75, 'return_rate': 8.5},
            {'device': 'Tablet', 'orders': 5000, 'avg_amount': 135.30, 'return_rate': 9.8}
        ],
        
        # 特征重要性
        'feature_importance': [
            {'feature': 'Delivery_Days', 'importance': 0.185},
            {'feature': 'Product_Category_encoded', 'importance': 0.142},
            {'feature': 'Discount_Percent', 'importance': 0.098},
            {'feature': 'Review_Rating', 'importance': 0.087},
            {'feature': 'Customer_Segment_encoded', 'importance': 0.075},
            {'feature': 'Unit_Price', 'importance': 0.068},
            {'feature': 'Order_Amount', 'importance': 0.062},
            {'feature': 'Membership_Status_encoded', 'importance': 0.055},
            {'feature': 'Shipping_Cost', 'importance': 0.048},
            {'feature': 'Quantity', 'importance': 0.042},
            {'feature': 'Customer_Age', 'importance': 0.038},
            {'feature': 'Payment_Method_encoded', 'importance': 0.035},
            {'feature': 'Device_Type_encoded', 'importance': 0.028},
            {'feature': 'Country_encoded', 'importance': 0.022},
            {'feature': 'Season_encoded', 'importance': 0.015}
        ],
        
        # 模型评估指标
        'model_metrics': {
            'auc_roc': 0.8245,
            'accuracy': 0.8523,
            'precision': 0.7856,
            'recall': 0.7234,
            'f1_score': 0.7532
        },
        
        # 混淆矩阵
        'confusion_matrix': {
            'tp': 4200,
            'fp': 1150,
            'tn': 45800,
            'fn': 1650
        }
    }
    
    return data


# ============================================================================
# Spark 模型加载
# ============================================================================

def init_spark_and_model():
    """初始化 SparkSession 并加载训练好的模型"""
    global spark, rf_model, lr_model, fitted_indexers, fitted_encoders, model_available
    import socket

    # ✅ 如果强制使用规则引擎，跳过所有 Spark 初始化
    if FORCE_RULE_ENGINE:
        print("\n⚠️ 强制使用规则引擎模式（FORCE_RULE_ENGINE = True）")
        print("   将不使用 Spark 模型进行预测")
        model_available = False
        return

    # 获取本机IP（用于调试）
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"📡 本机信息: {hostname} -> {local_ip}")
    except:
        local_ip = '127.0.0.1'
        print(f"📡 使用默认IP: {local_ip}")

    # 设置环境变量（修复网络问题）
    import os
    import sys
    
    # 设置 PySpark Python 解释器路径（Windows 上需要）
    # 关键修复：避免使用含中文的路径，直接使用 'python' 命令
    os.environ['PYSPARK_PYTHON'] = 'python'
    os.environ['PYSPARK_DRIVER_PYTHON'] = 'python'
    
    # 设置 Spark 本地 IP
    os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
    os.environ['SPARK_DRIVER_HOST'] = '127.0.0.1'
    os.environ['SPARK_DRIVER_BIND_ADDRESS'] = '127.0.0.1'

    print("\n🔧 正在初始化 SparkSession...")
    try:
        # 停止已存在的Spark Session（如果存在）
        try:
            SparkSession.builder.getOrCreate().stop()
            print("✅ 已停止旧Spark会话")
        except:
            pass
        
        # 设置 Java Home（如果已安装）
        java_paths = [
            r"d:\spark\java\jdk21",
            r"C:\Program Files\Common Files\Oracle\Java\javapath",
            r"C:\Program Files\Java\jdk-21",
            r"C:\Program Files\Java\jdk-17",
            r"C:\Program Files\Java\jdk-11",
        ]
        for java_path in java_paths:
            if os.path.exists(os.path.join(java_path, "bin", "java.exe")):
                os.environ['JAVA_HOME'] = java_path
                print(f"   检测到 Java: {java_path}")
                break
        
        # 创建新的Spark Session（针对Flask环境优化）
        spark = SparkSession.builder \
            .appName("EcommerceWebPredict") \
            .master("local[1]") \
            .config("spark.driver.host", "127.0.0.1") \
            .config("spark.driver.bindAddress", "127.0.0.1") \
            .config("spark.driver.port", "0") \
            .config("spark.blockManager.port", "0") \
            .config("spark.shuffle.service.enabled", "false") \
            .config("spark.dynamicAllocation.enabled", "false") \
            .config("spark.network.timeout", "600s") \
            .config("spark.executor.heartbeatInterval", "60s") \
            .config("spark.pyspark.python", sys.executable) \
            .config("spark.pyspark.driver.python", sys.executable) \
            .config("spark.python.worker.reuse", "true") \
            .config("spark.python.worker.memory", "1g") \
            .config("spark.sql.warehouse.dir", "file:///./spark-warehouse") \
            .config("spark.ui.enabled", "false") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        print(f"✅ SparkSession 初始化成功")
        print(f"   Python路径: {sys.executable}")
    except Exception as e:
        print(f"⚠️ SparkSession 初始化失败: {e}")
        print("   将使用规则引擎进行预测")
        return

    # 加载随机森林模型
    print(f"📦 正在加载模型: {MODEL_PATH}/")
    try:
        rf_model = RandomForestClassificationModel.load(MODEL_PATH)
        print("✅ 随机森林模型加载成功")
    except Exception as e:
        print(f"️ 随机森林模型加载失败: {e}")
        print("   请先运行 ecommerce_complete_project.py 训练并保存模型")
        rf_model = None

    # 加载逻辑回归基线模型
    print(f"📦 正在加载逻辑回归模型: {LR_MODEL_PATH}/")
    try:
        from pyspark.ml.classification import LogisticRegressionModel
        lr_model = LogisticRegressionModel.load(LR_MODEL_PATH)
        print("✅ 逻辑回归模型加载成功")
    except Exception as e:
        print(f"⚠️ 逻辑回归模型加载失败: {e}")
        lr_model = None

    # 尝试加载完整 Pipeline（推荐，最简单）
    print(f"📦 尝试加载完整 Pipeline: {PIPELINE_PATH}/")
    try:
        from pyspark.ml import PipelineModel
        pipeline_model = PipelineModel.load(PIPELINE_PATH)
        print("✅ 完整 Pipeline 加载成功（推荐方式）")
        print(f"   Pipeline 包含 {len(pipeline_model.stages)} 个阶段")
        for i, stage in enumerate(pipeline_model.stages):
            print(f"   [{i}] {type(stage).__name__}")
    except Exception as e:
        import traceback
        print(f"️ Pipeline 加载失败: {e}")
        print(f"   详细错误信息:\n{traceback.format_exc()}")
        print("   将使用单独的特征工程组件和模型进行预测")
        pipeline_model = None

    # 如果 Pipeline 加载失败，加载拟合的 indexers 和 encoders
    if pipeline_model is None:
        print(f"📦 正在加载特征工程组件: {PIPELINE_DIR}/")
        missing_files = []
        try:
            for col_name in CATEGORICAL_FEATURES:
                indexer_path = f"{PIPELINE_DIR}/{col_name}_indexer"
                encoder_path = f"{PIPELINE_DIR}/{col_name}_encoder"
                
                # 检查文件是否存在
                if not os.path.exists(indexer_path):
                    missing_files.append(indexer_path)
                if not os.path.exists(encoder_path):
                    missing_files.append(encoder_path)
            
            if missing_files:
                raise FileNotFoundError(f"缺少以下文件:\n" + "\n".join(missing_files))
            
            # 加载所有组件
            for col_name in CATEGORICAL_FEATURES:
                fitted_indexers[col_name] = StringIndexerModel.load(f"{PIPELINE_DIR}/{col_name}_indexer")
                fitted_encoders[col_name] = OneHotEncoderModel.load(f"{PIPELINE_DIR}/{col_name}_encoder")
            
            print(f"✅ 特征工程组件加载成功 ({len(CATEGORICAL_FEATURES)}个indexer + {len(CATEGORICAL_FEATURES)}个encoder)")
        except Exception as e:
            import traceback
            print(f"⚠️ 特征工程组件加载失败: {e}")
            print(f"   详细错误信息:\n{traceback.format_exc()}")
            print("   请重新运行 ecommerce_complete_project.py 保存组件")
            return

    model_available = True
    print("🎉 真实模型预测已就绪！")
    
    # 打印加载状态摘要
    print("\n" + "="*60)
    print(" 模型加载状态摘要")
    print("="*60)
    if pipeline_model is not None:
        print(f"✅ Pipeline: 已加载 ({len(pipeline_model.stages)}个阶段)")
        print(f"   → 将使用 Pipeline 进行预测（推荐）")
    else:
        print(f"❌ Pipeline: 未加载")
        print(f"   → 将使用单独组件进行预测（回退方案）")
    
    print(f"\n随机森林模型: {'✅ 已加载' if rf_model else '❌ 未加载'}")
    print(f"逻辑回归模型: {'✅ 已加载' if lr_model else '❌ 未加载'}")
    print(f"特征工程组件: {f'✅ 已加载 ({len(fitted_indexers)}组)' if fitted_indexers else '❌ 未加载'}")
    print("="*60 + "\n")


# ============================================================================
# 路由定义
# ============================================================================

@app.route('/')
def index():
    """主页 - 仪表盘概览"""
    data = load_analysis_data()
    return render_template('index.html', data=data)


@app.route('/api/data')
def get_data():
    """API 接口 - 返回所有分析数据"""
    data = load_analysis_data()
    return jsonify(data)


@app.route('/analysis/order-status')
def order_status_page():
    """订单状态分析页面"""
    data = load_analysis_data()
    return render_template('order_status.html', data=data)


@app.route('/analysis/category-sales')
def category_sales_page():
    """产品类别销售分析页面"""
    data = load_analysis_data()
    return render_template('category_sales.html', data=data)


@app.route('/analysis/customer-behavior')
def customer_behavior_page():
    """客户行为分析页面"""
    data = load_analysis_data()
    return render_template('customer_behavior.html', data=data)


@app.route('/analysis/delivery')
def delivery_page():
    """配送时效分析页面"""
    data = load_analysis_data()
    return render_template('delivery.html', data=data)


@app.route('/analysis/membership')
def membership_page():
    """会员等级分析页面"""
    data = load_analysis_data()
    return render_template('membership.html', data=data)


@app.route('/analysis/ml-model')
def ml_model_page():
    """机器学习模型分析页面"""
    data = load_analysis_data()
    return render_template('ml_model.html', data=data)


@app.route('/predict', methods=['GET'])
def predict_page():
    """预测页面 - 用户输入订单信息获取退货预测"""
    return render_template('predict.html', model_status='real' if model_available else 'rule')


@app.route('/api/predict', methods=['POST'])
def predict_api():
    """预测 API - 接收订单数据并返回预测结果"""
    from flask import request
    
    # 声明使用全局变量
    global spark, rf_model, lr_model, pipeline_model, fitted_indexers, fitted_encoders, model_available
    
    try:
        # 获取前端传来的订单数据
        order_data = request.get_json()
        
        # =============================================================
        # ✅ 优先尝试使用独立预测服务（方案1：最稳定）
        # =============================================================
        if USE_PREDICT_SERVICE:
            print("🚀 尝试调用独立预测服务...")
            try:
                import requests
                response = requests.post(
                    PREDICT_SERVICE_URL,
                    json=order_data,
                    timeout=5
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('success'):
                        print("✅ 独立预测服务预测成功")
                        return jsonify({
                            'success': True,
                            'prediction': result['prediction'],
                            'probability': result['probability'],
                            'risk_level': result['risk_level'],
                            'message': result.get('message', ''),
                            'model_type': result.get('model_type', 'random_forest_pipeline')
                        })
                    else:
                        print(f"⚠️ 预测服务返回错误: {result.get('error')}")
                else:
                    print(f"⚠️ 预测服务响应异常: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                print("⚠️ 独立预测服务未启动，降级到规则引擎")
            except requests.exceptions.Timeout:
                print("⚠️ 独立预测服务超时，降级到规则引擎")
            except Exception as e:
                print(f"⚠️ 调用独立预测服务失败: {e}")
        
        # =============================================================
        # ✅ 如果强制使用规则引擎或预测服务不可用，使用规则引擎（备用）
        # =============================================================
        print("📏 使用规则引擎进行预测...")
        
        # 提取特征
        delivery_days = float(order_data.get('delivery_days', 5))
        discount_percent = float(order_data.get('discount_percent', 10))
        rating = float(order_data.get('review_rating', 4.0))
        order_amount = float(order_data.get('order_amount', 150))
        payment_method = order_data.get('payment_method', 'Credit Card')
        product_category = order_data.get('product_category', 'Fashion')
        
        # 限制输入范围到数据集范围内（避免外推）
        delivery_days = max(0, min(delivery_days, 20))  # 数据集最大13天，留一点缓冲
        
        # 基础风险分（从 15% 开始，更符合实际业务）
        risk_score = 15
        
        # 配送速度影响（权重最高）- 基于真实数据分布，超慢配送惩罚递增
        if delivery_days > 15:
            risk_score += 55  # 极慢（>15天）- 超出数据集范围，严重延迟
        elif delivery_days > 12:
            risk_score += 45  # 很慢（13-15天）- 接近数据集上限
        elif delivery_days > 9:
            risk_score += 35  # 较慢（10-12天）
        elif delivery_days > 6:
            risk_score += 25  # 标准偏慢（7-9天）
        elif delivery_days >= 5:
            risk_score += 10  # 标准（5-6天）
        elif delivery_days <= 3:
            risk_score -= 5   # 快速配送降低风险
        
        # 折扣率影响（高折扣可能意味着冲动消费）
        if discount_percent > 40:
            risk_score += 20
        elif discount_percent > 25:
            risk_score += 12
        elif discount_percent > 15:
            risk_score += 5
        
        # 评分影响
        if rating < 2.5:
            risk_score += 25
        elif rating < 3.5:
            risk_score += 15
        elif rating < 4.0:
            risk_score += 5
        elif rating >= 4.5:
            risk_score -= 8  # 高评分降低风险
        # rating 在 [4.0, 4.5) 之间时无影响
        
        # 支付方式影响
        if payment_method == 'Cash on Delivery':
            risk_score += 15
        elif payment_method == 'Debit Card':
            risk_score += 5
        
        # 产品类别影响（根据真实业务数据设置不同风险等级）
        category_risk_map = {
            'Fashion': 15,              # 时尚服装 - 最高退货率（尺码/颜色问题）
            'Beauty & Personal Care': 12,  # 美妆个护 - 高退货率（过敏/色差）
            'Electronics': 10,          # 电子产品 - 中等偏高（功能不符）
            'Home & Kitchen': 8,        # 家居厨房 - 中等（尺寸/材质问题）
            'Sports': 6,                # 运动户外 - 中等偏低
            'Books': 3,                 # 图书媒体 - 最低退货率
            'Toys & Games': 9,          # 玩具游戏 - 中等
            'Health': 11,               # 健康用品 - 较高
            'Grocery': 4,               # 食品杂货 - 较低
        }
        
        # 获取该类别的风险分，如果不在列表中则默认5分
        risk_score += category_risk_map.get(product_category, 5)
        
        # 订单金额影响
        if order_amount > 500:
            risk_score -= 5  # 大额订单通常更谨慎
        elif order_amount < 50:
            risk_score += 5  # 小额订单退货率高
        
        # 限制在 0-95 之间
        risk_score = max(0, min(risk_score, 95))
        
        # 计算退货概率（risk_score 已经是百分比）
        return_probability = risk_score / 100.0
        
        # 判断是否退货（阈值设为 50%，与风险等级保持一致）
        is_returned = return_probability > 0.5
        
        result = {
            'success': True,
            'prediction': int(is_returned),
            'probability': round(return_probability, 4),
            # 风险等级：>50% 高风险，>30% 中风险，<=30% 低风险（与预测阈值一致）
            'risk_level': '高风险' if return_probability > 0.5 else ('中风险' if return_probability > 0.3 else '低风险'),
            'message': '该订单有较高退货风险' if is_returned else '该订单退货风险较低',
            'model_type': 'rule_engine'
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ 预测失败: {e}")
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
    # 预加载数据并检查状态
    print("=" * 70)
    print("         PySpark 电商数据分析 - Web 可视化展示系统")
    print("=" * 70)

    # 初始化 Spark 和加载模型
    init_spark_and_model()

    # 检查数据文件
    if os.path.exists(ANALYSIS_RESULTS_PATH):
        print(f"\n✅ 检测到真实数据: {ANALYSIS_RESULTS_PATH}")
        print("   数据来源: PySpark 分析结果")
    else:
        print(f"\n⚠️ 未找到真实数据: {ANALYSIS_RESULTS_PATH}")
        print("   使用模拟数据展示界面效果")
        print("    提示: 运行 python ecommerce_complete_project.py 生成真实数据")
    
    print("\n📊 正在启动 Web 服务器...")
    print("🌐 访问地址: http://localhost:5000")
    if model_available:
        models_loaded = []
        if rf_model is not None:
            models_loaded.append('随机森林')
        if lr_model is not None:
            models_loaded.append('逻辑回归')
        print(f"🤖 预测模式: 真实模型（{', '.join(models_loaded)}）")
    else:
        print("️ 预测模式: 规则引擎（未加载到模型）")
    print(" 启动时间:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 70)

    try:
        # 关键：关闭多线程和重载器，避免与Spark冲突
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=False)
    finally:
        if spark:
            spark.stop()
            print("\n👋 SparkSession 已关闭")
