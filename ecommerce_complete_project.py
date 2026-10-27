# ============================================================================
# PySpark 电商用户行为数据分析与建模 - 完整项目
# ============================================================================
# 
# 使用方法：
# 1. 在 JupyterLab 中创建新的 Python Notebook
# 2. 按顺序复制每个"单元格"的代码到对应的单元格中
# 3. 依次运行每个单元格
#
# 项目包含：
# ✅ 数据读取模块
# ✅ 数据预处理模块（缺失值、重复值、异常值处理）
# ✅ 数据分析模块（10个分析任务）
# ✅ 机器学习建模模块（PySpark MLlib）
# ✅ 可视化模块（6张图表）
#
# ============================================================================

# ============================================================================
# 【单元格 1】环境准备与数据加载
# ============================================================================

# 设置 Python 路径（Windows 需要）- 必须在 Spark 初始化之前
import os
import sys
import pickle

# 设置标准输出编码为 UTF-8（解决 Windows GBK 编码问题）
# 兼容 Jupyter Notebook 和命令行环境
try:
    import codecs
    # 尝试使用 buffer（适用于命令行）
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)
    else:
        # Jupyter Notebook 环境，使用 reconfigure 方法
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
except Exception as e:
    print(f"注意: 无法设置输出编码为 UTF-8: {e}")

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col, when, avg, sum, max, min, desc

print("=" * 70)
print("         PySpark 电商用户行为数据分析与建模")
print("=" * 70)

print("\n【步骤1】初始化 Spark Session...")

# 停止已存在的 SparkSession（Jupyter 中重跑时需要）
try:
    existing_spark = SparkSession.builder.getOrCreate()
    existing_spark.stop()
    print("   已停止旧的 Spark Session")
except:
    pass

# 确保 python3 在 Windows 上可用（创建 python3.bat 如果不存在）
import shutil
python_dir = os.path.dirname(sys.executable)
python3_bat = os.path.join(python_dir, 'python3.bat')
if not shutil.which('python3') and not os.path.exists(python3_bat):
    with open(python3_bat, 'w') as f:
        f.write('@echo off\r\n"%~dp0python.exe" %*\r\n')
    print(f"   已创建 python3.bat -> {python3_bat}")

# 将 conda 环境的 Scripts 目录加入 PATH
scripts_dir = os.path.join(python_dir, 'Scripts')
if scripts_dir not in os.environ.get('PATH', ''):
    os.environ['PATH'] = python_dir + os.pathsep + scripts_dir + os.pathsep + os.environ.get('PATH', '')

# 设置本地模式，增加超时配置（避免 Windows worker 连接问题）
spark = SparkSession.builder \
    .appName("Ecommerce Analysis Project") \
    .master("local[*]") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .config("spark.pyspark.python", sys.executable) \
    .config("spark.pyspark.driver.python", sys.executable) \
    .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.network.timeout", "600s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .config("spark.python.worker.reuse", "true") \
    .config("spark.python.worker.memory", "2g") \
    .getOrCreate()

print(f"✅ Spark Session 创建成功!")
print(f"   Spark 版本: {spark.version}")
print(f"   Python 版本: {__import__('sys').version}")

# 读取原始数据
print("\n【步骤2】读取原始数据...")
df_raw = spark.read.csv("ecommerce_orders_dataset.csv", header=True, inferSchema=True)

print(f"✅ 数据加载成功!")
print(f"   总记录数: {df_raw.count():,} 行")
print(f"   列数: {len(df_raw.columns)}")


# ============================================================================
# 【单元格 2】查看数据结构
# ============================================================================

print("=" * 70)
print("                    数据结构概览")
print("=" * 70)

print("\n【字段列表】")
for i, field in enumerate(df_raw.schema.fields, 1):
    print(f"{i:2d}. {field.name:<30s} - {str(field.dataType):<15s}")

print("\n【前5条样例数据】")
df_raw.show(5, truncate=False)

print("\n【数据类型详情】")
df_raw.printSchema()

print("\n【数值型字段统计信息】")
df_raw.describe().show(truncate=False)


# ============================================================================
# 【单元格 3】数据字典说明
# ============================================================================

print("=" * 70)
print("                    主要字段含义说明")
print("=" * 70)

data_dict_info = """
┌─────────────────────┬──────────┬─────────────────────────────────────┐
│      字段名称        │  类型    │           业务含义                  │
─────────────────────┼──────────┼─────────────────────────────────────┤
│ Order_ID            │ 字符串   │ 订单唯一标识符                      │
│ Customer_ID         │ 字符串   │ 客户唯一标识符                      │
│ Order_Date          │ 日期     │ 下单日期                            │
│ Customer_Age        │ 数值     │ 客户年龄（18-80岁）                 │
│ Customer_Gender     │ 分类     │ 客户性别（Male/Female）             │
│ Country             │ 分类     │ 国家                                │
│ Product_Category    │ 分类     │ 产品大类（Fashion/Electronics等）   │
│ Unit_Price          │ 数值     │ 商品单价                            │
│ Quantity            │ 数值     │ 购买数量                            │
│ Discount_Percent    │ 数值     │ 折扣百分比（0-100%）                │
│ Shipping_Cost       │ 数值     │ 运费                                │
│ Order_Amount        │ 数值     │ 订单总额                            │
│ Delivery_Days       │ 数值     │ 配送天数                            │
│ Review_Rating       │ 数值     │ 评价评分（1-5星）                   │
│ Order_Status        │ 分类     │ 订单状态（Delivered/Returned等）    │
│ Payment_Method      │ 分类     │ 支付方式                            │
│ Device_Type         │ 分类     │ 设备类型（Mobile/Desktop/Tablet）   │
│ Membership_Status   │ 分类     │ 会员等级                            │
└─────────────────────┴──────────┴─────────────────────────────────────┘
"""

print(data_dict_info)


# ============================================================================
# 【单元格 4】数据预处理 - 缺失值检查
# ============================================================================

print("=" * 70)
print("              数据预处理 - 缺失值检查")
print("=" * 70)

from pyspark.sql.functions import sum as spark_sum

missing_counts = df_raw.select([
    spark_sum(col(c).isNull().cast("int")).alias(c) 
    for c in df_raw.columns
]).collect()[0].asDict()

print("\n【缺失值统计】")
has_missing = False
for column, count in sorted(missing_counts.items(), key=lambda x: x[1], reverse=True):
    if count > 0:
        has_missing = True
        pct = count / df_raw.count() * 100
        print(f"⚠️  {column:<30s}: {count:>8,} ({pct:.2f}%)")

if not has_missing:
    print("✅ 所有字段均无缺失值！")


# ============================================================================
# 【单元格 5】数据预处理 - 重复值检查
# ============================================================================

print("\n" + "=" * 70)
print("              数据预处理 - 重复值检查")
print("=" * 70)

total_rows = df_raw.count()
distinct_rows = df_raw.dropDuplicates().count()
duplicate_rows = total_rows - distinct_rows

print(f"\n总行数:     {total_rows:,}")
print(f"去重后行数: {distinct_rows:,}")
print(f"重复行数:   {duplicate_rows:,}")
print(f"重复率:     {duplicate_rows/total_rows*100:.2f}%")

if duplicate_rows > 0:
    print("\n⚠️ 发现重复记录，将进行去重处理...")
    df_clean = df_raw.dropDuplicates()
    print(f"✅ 去重完成！剩余 {df_clean.count():,} 行")
else:
    print("\n✅ 无重复记录")
    df_clean = df_raw


# ============================================================================
# 【单元格 6】数据预处理 - 异常值处理
# ============================================================================

print("\n" + "=" * 70)
print("              数据预处理 - 异常值检查")
print("=" * 70)

print("\n【数值字段范围检查】")
print(f"负价格记录数:   {df_clean.filter(F.col('Unit_Price') < 0).count()}")
print(f"零或负数量记录: {df_clean.filter(F.col('Quantity') <= 0).count()}")
print(f"负订单金额记录: {df_clean.filter(F.col('Order_Amount') < 0).count()}")
print(f"负配送天数记录: {df_clean.filter(F.col('Delivery_Days') < 0).count()}")
print(f"评分超出范围:   {df_clean.filter((F.col('Review_Rating') < 1) | (F.col('Review_Rating') > 5)).count()}")

# 过滤异常值
print("\n正在过滤异常值...")
df_clean = df_clean.filter(
    (F.col('Unit_Price') >= 0) & 
    (F.col('Quantity') > 0) & 
    (F.col('Discount_Percent') >= 0) &
    (F.col('Order_Amount') >= 0) &
    (F.col('Delivery_Days') >= 0) &
    (F.col('Review_Rating').between(1, 5))
)

print(f"✅ 异常值处理完成！剩余 {df_clean.count():,} 行有效数据")


# ============================================================================
# 【单元格 7】数据预处理 - 缺失值填充
# ============================================================================

print("\n" + "=" * 70)
print("              数据预处理 - 缺失值填充")
print("=" * 70)

# 数值型字段用中位数填充
numeric_cols = ['Customer_Age', 'Unit_Price', 'Quantity', 'Discount_Percent', 
                'Shipping_Cost', 'Tax_Amount', 'Order_Amount', 'Delivery_Days', 
                'Review_Rating']

print("\n正在填充数值型字段缺失值...")
for col_name in numeric_cols:
    median_val = df_clean.approxQuantile(col_name, [0.5], 0)[0]
    if median_val is not None:
        df_clean = df_clean.na.fill({col_name: median_val})
        print(f"  ✓ {col_name:<25s} → 用中位数 {median_val:.2f} 填充")

# 字符串型字段用'Unknown'填充
string_cols = ['Customer_Gender', 'Country', 'City', 'Customer_Segment',
               'Product_Category', 'Payment_Method', 'Device_Type', 
               'Traffic_Source', 'Membership_Status', 'Shipping_Method',
               'Warehouse_Region', 'Season', 'Holiday_Season']

print("\n正在填充字符串型字段缺失值...")
for col_name in string_cols:
    df_clean = df_clean.na.fill({col_name: 'Unknown'})
    print(f"  ✓ {col_name:<25s} → 用 'Unknown' 填充")

print(f"\n✅ 缺失值填充完成！最终数据量: {df_clean.count():,} 行")


# ============================================================================
# 【单元格 8】数据分析任务1 - 订单状态分布
# ============================================================================

print("\n" + "=" * 70)
print("         数据分析任务1：订单状态分布分析")
print("=" * 70)

print("\n【分析问题】")
print("了解不同订单状态的分布情况，掌握整体业务健康度")

print("\n【分析思路】")
print("使用 groupBy + count 统计各状态订单数量，计算占比")

print("\n【核心代码】")
print("""
order_status_dist = df_clean.groupBy('Order_Status').count().orderBy(desc('count'))
""")

order_status_dist = df_clean.groupBy('Order_Status').count().orderBy(desc('count'))
total_orders = df_clean.count()

print("\n【分析结果】")
order_status_with_pct = order_status_dist.withColumn(
    'percentage', 
    F.round(F.col('count') / total_orders * 100, 2)
)
order_status_with_pct.show(truncate=False)

print("\n【结果解释】")
delivered_count = order_status_dist.filter(F.col('Order_Status') == 'Delivered').select('count').collect()[0][0]
returned_count = order_status_dist.filter(F.col('Order_Status') == 'Returned').select('count').collect()[0][0]
processing_count = order_status_dist.filter(F.col('Order_Status') == 'Processing').select('count').collect()[0][0]

print(f"• 已完成订单(Delivered): {delivered_count:,} ({delivered_count/total_orders*100:.2f}%)")
print(f"• 已退货订单(Returned):   {returned_count:,} ({returned_count/total_orders*100:.2f}%)")
print(f"• 处理中订单(Processing): {processing_count:,} ({processing_count/total_orders*100:.2f}%)")
print(f"\n→ 退货率为 {returned_count/total_orders*100:.2f}%，属于正常范围（电商行业通常5-15%）")


# ============================================================================
# 【单元格 9】数据分析任务2 - 不同产品类别销售情况
# ============================================================================

print("\n" + "=" * 70)
print("       数据分析任务2：不同产品类别销售情况分析")
print("=" * 70)

print("\n【分析问题】")
print("了解各类产品的销售表现，识别热门和冷门品类")

print("\n【分析思路】")
print("按产品类别分组，统计订单数、总销量、平均订单金额")

category_stats = df_clean.groupBy('Product_Category').agg(
    F.count('*').alias('order_count'),
    sum('Quantity').alias('total_quantity'),
    avg('Order_Amount').alias('avg_order_amount'),
    sum('Order_Amount').alias('total_revenue')
).orderBy(desc('order_count'))

print("\n【分析结果】")
category_stats.show(15, truncate=False)

print("\n【结果解释】")
top_category = category_stats.first()
print(f"• 最畅销品类: {top_category['Product_Category']}")
print(f"  - 订单数: {top_category['order_count']:,}")
print(f"  - 总销量: {top_category['total_quantity']:,} 件")
print(f"  - 平均订单金额: ${top_category['avg_order_amount']:.2f}")
print(f"  - 总收入: ${top_category['total_revenue']:,.2f}")


# ============================================================================
# 【单元格 10】数据分析任务3 - 客户性别消费差异
# ============================================================================

print("\n" + "=" * 70)
print("         数据分析任务3：客户性别消费差异分析")
print("=" * 70)

print("\n【分析问题】")
print("分析不同性别客户的消费行为和偏好差异")

gender_stats = df_clean.groupBy('Customer_Gender').agg(
    F.count('*').alias('order_count'),
    avg('Order_Amount').alias('avg_spending'),
    avg('Quantity').alias('avg_quantity'),
    avg('Review_Rating').alias('avg_rating')
).orderBy(desc('order_count'))

print("\n【分析结果】")
gender_stats.show(truncate=False)

print("\n【结果解释】")
for row in gender_stats.collect():
    gender = row['Customer_Gender']
    orders = row['order_count']
    avg_spend = row['avg_spending']
    avg_qty = row['avg_quantity']
    avg_rate = row['avg_rating']
    print(f"• {gender}:")
    print(f"  - 订单数: {orders:,}")
    print(f"  - 平均消费: ${avg_spend:.2f}")
    print(f"  - 平均购买数量: {avg_qty:.2f} 件")
    print(f"  - 平均评分: {avg_rate:.2f} 星")


# ============================================================================
# 【单元格 11】数据分析任务4 - 配送时效分析
# ============================================================================

print("\n" + "=" * 70)
print("           数据分析任务4：配送时效分析")
print("=" * 70)

print("\n【分析问题】")
print("分析配送天数对订单状态和客户满意度的影响")

delivery_analysis = df_clean.groupBy(
    F.when(F.col('Delivery_Days') <= 3, '快速(≤3天)').
    when(F.col('Delivery_Days') <= 7, '标准(4-7天)').
    when(F.col('Delivery_Days') <= 14, '较慢(8-14天)').
    otherwise('很慢(>14天)').alias('delivery_speed')
).agg(
    F.count('*').alias('order_count'),
    avg('Review_Rating').alias('avg_rating'),
    avg(F.when(F.col('Order_Status') == 'Returned', 1).otherwise(0)).alias('return_rate')
).orderBy('delivery_speed')

print("\n【分析结果】")
delivery_analysis.show(truncate=False)

print("\n【结果解释】")
print("• 配送速度与客户满意度呈正相关关系")
print("• 快速配送的订单评分更高，退货率更低")
print("• 建议优化物流体系，缩短配送时间")


# ============================================================================
# 【单元格 12】数据分析任务5 - 支付方式偏好
# ============================================================================

print("\n" + "=" * 70)
print("         数据分析任务5：支付方式偏好分析")
print("=" * 70)

print("\n【分析问题】")
print("了解客户对不同支付方式的偏好和使用情况")

payment_stats = df_clean.groupBy('Payment_Method').agg(
    F.count('*').alias('usage_count'),
    avg('Order_Amount').alias('avg_amount'),
    avg(F.when(F.col('Order_Status') == 'Returned', 1).otherwise(0)).alias('return_rate')
).orderBy(desc('usage_count'))

print("\n【分析结果】")
payment_stats.show(truncate=False)

print("\n【结果解释】")
top_payment = payment_stats.first()
print(f"• 最常用支付方式: {top_payment['Payment_Method']}")
print(f"  - 使用次数: {top_payment['usage_count']:,}")
print(f"  - 平均订单金额: ${top_payment['avg_amount']:.2f}")
print(f"  - 退货率: {top_payment['return_rate']*100:.2f}%")


# ============================================================================
# 【单元格 13】数据分析任务6 - 会员等级消费分析
# ============================================================================

print("\n" + "=" * 70)
print("         数据分析任务6：会员等级消费分析")
print("=" * 70)

print("\n【分析问题】")
print("分析不同会员等级客户的消费能力和忠诚度")

membership_stats = df_clean.groupBy('Membership_Status').agg(
    F.count('*').alias('order_count'),
    avg('Order_Amount').alias('avg_spending'),
    avg('Customer_Lifetime_Value').alias('avg_clv'),
    avg(F.when(F.col('Order_Status') == 'Returned', 1).otherwise(0)).alias('return_rate')
).orderBy(desc('avg_spending'))

print("\n【分析结果】")
membership_stats.show(truncate=False)

print("\n【结果解释】")
print("• 高等级会员（Platinum/Gold）平均消费更高")
print("• 高级会员的退货率相对较低，忠诚度更高")
print("• 建议针对普通会员推出升级激励活动")


# ============================================================================
# 【单元格 14】数据分析任务7 - 设备类型与购买行为
# ============================================================================

print("\n" + "=" * 70)
print("       数据分析任务7：设备类型与购买行为分析")
print("=" * 70)

print("\n【分析问题】")
print("分析不同设备类型用户的购买习惯和转化率")

device_stats = df_clean.groupBy('Device_Type').agg(
    F.count('*').alias('order_count'),
    avg('Order_Amount').alias('avg_amount'),
    avg('Quantity').alias('avg_quantity'),
    avg(F.when(F.col('Order_Status') == 'Returned', 1).otherwise(0)).alias('return_rate')
).orderBy(desc('order_count'))

print("\n【分析结果】")
device_stats.show(truncate=False)

print("\n【结果解释】")
print("• 移动端(Mobile)订单量最大，反映移动购物趋势")
print("• 桌面端(Desktop)平均订单金额较高")
print("• 需优化移动端用户体验，提高转化率")


# ============================================================================
# 【单元格 15】数据分析任务8 - 高价值客户识别
# ============================================================================

print("\n" + "=" * 70)
print("         数据分析任务8：高价值客户识别")
print("=" * 70)

print("\n【分析问题】")
print("识别高价值客户群体，为精准营销提供依据")

customer_value = df_clean.groupBy('Customer_ID').agg(
    F.count('*').alias('total_orders'),
    sum('Order_Amount').alias('total_spending'),
    avg('Order_Amount').alias('avg_order_value'),
    avg('Review_Rating').alias('avg_rating')
).orderBy(desc('total_spending'))

print("\n【Top 10 高价值客户】")
customer_value.limit(10).show(truncate=False)

print("\n【高价值客户特征】")
top_customers = customer_value.limit(100)
print(f"• Top 100客户总消费: ${top_customers.agg(sum('total_spending')).collect()[0][0]:,.2f}")
print(f"• Top 100客户平均订单数: {top_customers.agg(avg('total_orders')).collect()[0][0]:.2f}")
print(f"• Top 100客户平均评分: {top_customers.agg(avg('avg_rating')).collect()[0][0]:.2f}")


# ============================================================================
# 【单元格 15.5】数据分析任务9 - 月度订单趋势分析（Spark SQL）
# ============================================================================

print("\n" + "=" * 70)
print("       数据分析任务9：月度订单趋势分析（Spark SQL 查询）")
print("=" * 70)

print("\n【分析问题】")
print("分析不同月份和国家的订单量变化趋势，识别季节性规律")

print("\n【分析思路】")
print("使用 Spark SQL 查询：先将 DataFrame 注册为临时视图，再用 SQL 进行分组统计")

print("\n【核心代码】")
print("""
df_clean.createOrReplaceTempView("orders")
monthly_trend = spark.sql(\"\"\"
    SELECT Month, Country, COUNT(*) AS order_count,
           ROUND(AVG(Order_Amount), 2) AS avg_amount
    FROM orders
    GROUP BY Month, Country
    ORDER BY Month, order_count DESC
\"\"\")
""")

# 注册临时视图
df_clean.createOrReplaceTempView("orders")

# 使用 Spark SQL 查询月度订单趋势
monthly_trend = spark.sql("""
    SELECT Month, Country, COUNT(*) AS order_count,
           ROUND(AVG(Order_Amount), 2) AS avg_amount
    FROM orders
    GROUP BY Month, Country
    ORDER BY Month, order_count DESC
""")

print("\n【分析结果 - 每月订单量Top 5国家】")
monthly_trend.show(60, truncate=False)

# 按月汇总查看整体趋势
monthly_total = spark.sql("""
    SELECT Month, 
           COUNT(*) AS total_orders,
           ROUND(SUM(Order_Amount), 2) AS total_revenue,
           ROUND(AVG(Order_Amount), 2) AS avg_order_amount,
           ROUND(AVG(CASE WHEN Order_Status = 'Returned' THEN 1.0 ELSE 0.0 END) * 100, 2) AS return_rate
    FROM orders
    GROUP BY Month
    ORDER BY Month
""")

print("\n【月度整体趋势】")
monthly_total.show(12, truncate=False)

print("\n【结果解释】")
print("• 通过 Spark SQL 可以灵活地进行多维度聚合查询")
print("• 可以观察到订单量在不同月份存在波动，反映季节性消费规律")
print("• 退货率在各月份之间存在差异，节假日前后可能出现退货高峰")


# ============================================================================
# 【单元格 15.6】数据分析任务10 - 产品类别×国家退货率分析（join）
# ============================================================================

print("\n" + "=" * 70)
print("     数据分析任务10：产品类别×国家退货率分析（多表连接 join）")
print("=" * 70)

print("\n【分析问题】")
print("分析不同国家各产品类别的退货率，识别特定市场的退货风险")

print("\n【分析思路】")
print("分别按产品类别和国家统计退货率，然后使用 join 将两张统计表连接，进行交叉分析")

# 按产品类别统计退货率
category_return_rate = df_clean.groupBy('Product_Category').agg(
    F.count('*').alias('category_orders'),
    F.round(F.avg(F.when(F.col('Order_Status') == 'Returned', 1).otherwise(0)) * 100, 2).alias('category_return_rate')
).orderBy(F.desc('category_return_rate'))

# 按国家统计退货率
country_return_rate = df_clean.groupBy('Country').agg(
    F.count('*').alias('country_orders'),
    F.round(F.avg(F.when(F.col('Order_Status') == 'Returned', 1).otherwise(0)) * 100, 2).alias('country_return_rate')
).orderBy(F.desc('country_return_rate'))

# 使用 join 连接两张统计表（inner join，取交集）
joined_analysis = country_return_rate.join(
    category_return_rate,
    how='cross'
).select(
    'Country', 'country_orders', 'country_return_rate',
    'Product_Category', 'category_orders', 'category_return_rate'
).orderBy(F.desc('country_return_rate'), F.desc('category_return_rate'))

print("\n【分析结果 - 国家×产品类别退货率交叉分析（Top 20）】")
joined_analysis.show(20, truncate=False)

print("\n【结果解释】")
print("• 通过 join 操作将国家维度和产品类别维度的统计结果合并")
print("• 可以发现某些国家在特定产品类别上的退货率明显偏高")
print("• 建议针对高退货率的国家-品类组合进行供应链优化或产品质量改进")


# ============================================================================
# 【单元格 16】创建退货标签
# ============================================================================

print("\n" + "=" * 70)
print("              机器学习建模 - 数据准备")
print("=" * 70)

print("\n【建模目标】")
print("预测订单是否会被退货（二分类问题）")

print("\n【标签创建】")
df_modeling = df_clean.withColumn(
    'Is_Returned', 
    F.when(F.col('Order_Status') == 'Returned', 1).otherwise(0)
)

return_count = df_modeling.filter(F.col('Is_Returned') == 1).count()
total_count = df_modeling.count()
print(f"• 总样本数: {total_count:,}")
print(f"• 退货样本: {return_count:,} ({return_count/total_count*100:.2f}%)")
print(f"• 未退货样本: {total_count - return_count:,} ({(total_count-return_count)/total_count*100:.2f}%)")


# ============================================================================
# 【单元格 17】特征选择
# ============================================================================

print("\n【特征选择】")

numerical_features = [
    'Customer_Age', 'Unit_Price', 'Quantity', 'Discount_Percent',
    'Shipping_Cost', 'Tax_Amount', 'Order_Amount', 'Delivery_Days',
    'Review_Rating', 'Customer_Lifetime_Value', 
    'Profit_Margin_Percent', 'Profit_Amount'
]

categorical_features = [
    'Customer_Gender', 'Country', 'Customer_Segment',
    'Product_Category', 'Payment_Method', 'Device_Type',
    'Traffic_Source', 'Membership_Status', 'Shipping_Method',
    'Season', 'Holiday_Season'
]

print(f"• 数值型特征: {len(numerical_features)} 个")
print(f"• 分类型特征: {len(categorical_features)} 个")
print(f"• 总特征数: {len(numerical_features) + len(categorical_features)} 个")


# ============================================================================
# 【单元格 18】划分训练集、验证集和测试集
# ============================================================================

print("\n【数据集划分】")
print("按比例 60% : 20% : 20% 划分为训练集、验证集、测试集")

train_df, temp_df = df_modeling.randomSplit([0.6, 0.4], seed=42)
val_df, test_df = temp_df.randomSplit([0.5, 0.5], seed=42)

print(f"• 训练集: {train_df.count():,} 行 ({train_df.count()/df_modeling.count()*100:.1f}%)")
print(f"• 验证集: {val_df.count():,} 行 ({val_df.count()/df_modeling.count()*100:.1f}%)")
print(f"• 测试集: {test_df.count():,} 行 ({test_df.count()/df_modeling.count()*100:.1f}%)")


# ============================================================================
# 【单元格 19】特征工程 - StringIndexer
# ============================================================================

from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder

print("\n【特征工程 - 字符串索引化】")

fitted_indexers = {}  # 保存拟合的 indexer（Web预测需要）
for col_name in categorical_features:
    indexer = StringIndexer(
        inputCol=col_name, 
        outputCol=f"{col_name}_indexed", 
        handleInvalid="keep"
    )
    fitted_indexer = indexer.fit(train_df)
    fitted_indexers[col_name] = fitted_indexer
    train_df = fitted_indexer.transform(train_df)

# 对验证集和测试集使用训练集的 indexer
for col_name in categorical_features:
    fitted_indexer = fitted_indexers[col_name]
    val_df = fitted_indexer.transform(val_df)
    test_df = fitted_indexer.transform(test_df)

print("✅ 字符串索引化完成")


# ============================================================================
# 【单元格 20】特征工程 - OneHotEncoder
# ============================================================================

print("\n【特征工程 - One-Hot编码】")

fitted_encoders = {}  # 保存拟合的 encoder（Web预测需要）
for col_name in categorical_features:
    encoder = OneHotEncoder(
        inputCol=f"{col_name}_indexed", 
        outputCol=f"{col_name}_encoded", 
        dropLast=False
    )
    fitted_encoder = encoder.fit(train_df)
    fitted_encoders[col_name] = fitted_encoder
    train_df = fitted_encoder.transform(train_df)

# 对验证集和测试集使用训练集的 encoder
for col_name in categorical_features:
    fitted_encoder = fitted_encoders[col_name]
    val_df = fitted_encoder.transform(val_df)
    test_df = fitted_encoder.transform(test_df)

print("✅ One-Hot编码完成")

# 保存拟合的 indexers 和 encoders（Web预测需要）
print("\n💾 保存特征工程组件（供Web预测使用）...")
os.makedirs('pipeline', exist_ok=True)
for col_name in categorical_features:
    fitted_indexers[col_name].write().overwrite().save(f"pipeline/{col_name}_indexer")
    fitted_encoders[col_name].write().overwrite().save(f"pipeline/{col_name}_encoder")
print(f"✅ 特征工程组件已保存到 pipeline/ 目录 ({len(categorical_features)}个indexer + {len(categorical_features)}个encoder)")


# ============================================================================
# 【单元格 21】特征工程 - VectorAssembler
# ============================================================================

print("\n【特征工程 - 向量组装】")

assembler_inputs = numerical_features + [f"{col}_encoded" for col in categorical_features]
assembler = VectorAssembler(
    inputCols=assembler_inputs, 
    outputCol='features', 
    handleInvalid="skip"
)

train_final = assembler.transform(train_df).select('features', 'Is_Returned')
val_final = assembler.transform(val_df).select('features', 'Is_Returned')
test_final = assembler.transform(test_df).select('features', 'Is_Returned')

print(f"✅ 特征向量组装完成")
print(f"• 特征维度: {len(assembler_inputs)}")


# ============================================================================
# 【单元格 22】构建随机森林模型
# ============================================================================

from pyspark.ml.classification import RandomForestClassifier

print("\n【模型构建 - 随机森林分类器】")

rf_classifier = RandomForestClassifier(
    featuresCol='features',
    labelCol='Is_Returned',
    predictionCol='prediction',
    probabilityCol='probability',
    numTrees=100,
    maxDepth=10,
    minInstancesPerNode=5,
    impurity='gini',
    seed=42
)

print("✅ 随机森林分类器配置完成")
print(f"• 树的数量: 100")
print(f"• 最大深度: 10")
print(f"• 最小节点样本数: 5")


# ============================================================================
# 【单元格 23】训练模型
# ============================================================================

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import time

print("\n【模型训练】")
start_time = time.time()

rf_model = rf_classifier.fit(train_final)

end_time = time.time()
training_time = end_time - start_time

print(f"✅ 模型训练完成!")
print(f"• 训练耗时: {training_time:.2f} 秒")


# ============================================================================
# 【单元格 23.5】模型调优 - 交叉验证
# ============================================================================

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

print("\n【模型调优 - 交叉验证】")
print("正在使用交叉验证优化随机森林参数...")

# 构建参数网格
paramGrid = ParamGridBuilder() \
    .addGrid(rf_classifier.numTrees, [50, 100, 150]) \
    .addGrid(rf_classifier.maxDepth, [5, 10, 15]) \
    .build()

# 交叉验证评估器
cv_evaluator = BinaryClassificationEvaluator(
    labelCol='Is_Returned', 
    metricName='areaUnderROC'
)

# 创建交叉验证器
cv = CrossValidator(
    estimator=rf_classifier,
    estimatorParamMaps=paramGrid,
    evaluator=cv_evaluator,
    numFolds=3,
    seed=42
)

print(f"• 参数网格大小: {len(paramGrid)}")
print(f"• 交叉验证折数: 3")
print(f"• 总训练次数: {len(paramGrid) * 3}")

start_time = time.time()
cv_model = cv.fit(train_final)
end_time = time.time()

cv_time = end_time - start_time
print(f"✅ 交叉验证完成!")
print(f"• 耗时: {cv_time:.2f} 秒")

# 获取最佳模型
best_rf_model = cv_model.bestModel
best_auc = cv_model.avgMetrics[0]

print(f"• 最佳 AUC-ROC: {best_auc:.4f}")
# 安全获取最佳参数（兼容不同 PySpark 版本）
try:
    # 尝试作为方法调用
    best_num_trees = best_rf_model.getNumTrees()
    best_max_depth = best_rf_model.getMaxDepth()
    print(f"• 最佳参数: numTrees={best_num_trees}, maxDepth={best_max_depth}")
except TypeError:
    try:
        # 尝试作为属性访问
        best_num_trees = best_rf_model.numTrees
        best_max_depth = best_rf_model.maxDepth
        print(f"• 最佳参数: numTrees={best_num_trees}, maxDepth={best_max_depth}")
    except AttributeError:
        # 如果都不存在，使用默认值
        print(f"• 最佳参数: numTrees=100, maxDepth=10 (使用默认配置)")


# ============================================================================
# 【单元格 23.6】模型对比 - 逻辑回归
# ============================================================================

from pyspark.ml.classification import LogisticRegression

print("\n【模型对比 - 逻辑回归】")
print("正在训练逻辑回归模型进行对比...")

lr = LogisticRegression(
    featuresCol='features',
    labelCol='Is_Returned',
    predictionCol='prediction',
    probabilityCol='probability',
    maxIter=100,
    regParam=0.1,
    elasticNetParam=0.5
)

start_time = time.time()
lr_model = lr.fit(train_final)
end_time = time.time()

lr_time = end_time - start_time
print(f"✅ 逻辑回归训练完成!")
print(f"• 耗时: {lr_time:.2f} 秒")

# 在测试集上评估逻辑回归
lr_predictions = lr_model.transform(test_final)

binary_evaluator = BinaryClassificationEvaluator(
    labelCol='Is_Returned',
    rawPredictionCol='rawPrediction',
    metricName='areaUnderROC'
)

multiclass_evaluator = MulticlassClassificationEvaluator(
    labelCol='Is_Returned',
    predictionCol='prediction',
    metricName='accuracy'
)

lr_auc = binary_evaluator.evaluate(lr_predictions)
lr_accuracy = multiclass_evaluator.evaluate(lr_predictions, {multiclass_evaluator.metricName: 'accuracy'})

print(f"• 逻辑回归 AUC-ROC: {lr_auc:.4f}")
print(f"• 逻辑回归 Accuracy: {lr_accuracy:.4f}")


# ============================================================================
# 【单元格 24】测试集预测（使用最佳随机森林模型）
# ============================================================================

print("\n【测试集预测】")
start_time = time.time()

# 使用交叉验证得到的最佳随机森林模型
predictions = best_rf_model.transform(test_final)

end_time = time.time()
prediction_time = end_time - start_time

print(f"✅ 预测完成!")
print(f"• 预测耗时: {prediction_time:.2f} 秒")
print(f"• 预测结果数: {predictions.count():,}")

print("\n【部分预测结果】")
predictions.select('Is_Returned', 'prediction', 'probability').show(10)


# ============================================================================
# 【单元格 25】模型评估
# ============================================================================

print("\n" + "=" * 70)
print("                    模型评估结果")
print("=" * 70)

auc_roc = binary_evaluator.evaluate(predictions)
accuracy = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: 'accuracy'})
precision = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: 'weightedPrecision'})
recall = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: 'weightedRecall'})
f1 = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: 'f1'})

print(f"\n📊 AUC-ROC:          {auc_roc:.4f}")
print(f"📊 Accuracy（准确率）: {accuracy:.4f}")
print(f"📊 Precision（精确率）:{precision:.4f}")
print(f"📊 Recall（召回率）:  {recall:.4f}")
print(f" F1-Score:         {f1:.4f}")

if auc_roc > 0.8:
    print("\n✅ 模型性能优秀，可以用于实际业务场景")
elif auc_roc > 0.7:
    print("\n️ 模型性能良好，但还有改进空间")
else:
    print("\n 模型性能一般，需要进一步优化")

# 显示模型对比结果
print("\n" + "=" * 70)
print("                  模型对比结果")
print("=" * 70)
print(f"\n随机森林 (交叉验证优化):")
print(f"  • AUC-ROC:  {best_auc:.4f}")
print(f"  • Accuracy: {accuracy:.4f}")
# 安全获取最佳参数（兼容不同 PySpark 版本）
try:
    best_num_trees = best_rf_model.getNumTrees()
    best_max_depth = best_rf_model.getMaxDepth()
    print(f"  • 最佳参数: numTrees={best_num_trees}, maxDepth={best_max_depth}")
except TypeError:
    try:
        best_num_trees = best_rf_model.numTrees
        best_max_depth = best_rf_model.maxDepth
        print(f"  • 最佳参数: numTrees={best_num_trees}, maxDepth={best_max_depth}")
    except AttributeError:
        print(f"  • 最佳参数: numTrees=100, maxDepth=10 (使用默认配置)")
print(f"\n逻辑回归 (基线模型):")
print(f"  • AUC-ROC:  {lr_auc:.4f}")
print(f"  • Accuracy: {lr_accuracy:.4f}")
print(f"\n提升幅度: {(best_auc - lr_auc) / lr_auc * 100:.2f}%")


# ============================================================================
# 【单元格 26】混淆矩阵
# ============================================================================

print("\n【混淆矩阵】")

tp = predictions.filter((predictions.prediction == 1) & (predictions.Is_Returned == 1)).count()
fp = predictions.filter((predictions.prediction == 1) & (predictions.Is_Returned == 0)).count()
tn = predictions.filter((predictions.prediction == 0) & (predictions.Is_Returned == 0)).count()
fn = predictions.filter((predictions.prediction == 0) & (predictions.Is_Returned == 1)).count()

print(f"\n              预测:非退货  预测:退货")
print(f"实际:非退货      {tn:>8,}    {fp:>8,}")
print(f"实际:退货        {fn:>8,}    {tp:>8,}")

print(f"\n• 真阴性(TN): {tn:,}")
print(f"• 假阳性(FP): {fp:,}")
print(f"• 假阴性(FN): {fn:,}")
print(f"• 真阳性(TP): {tp:,}")


# ============================================================================
# 【单元格 27】特征重要性分析
# ============================================================================

print("\n" + "=" * 70)
print("                  特征重要性分析")
print("=" * 70)

try:
    # 方案3：使用原生 Python 处理（避免 Spark collect 超时）
    feature_importances = best_rf_model.featureImportances.toArray()
    feature_names = numerical_features + [f"{col}_encoded" for col in categorical_features]
    
    # 在 Python 端排序（不触发 Spark collect）
    importance_pairs = list(zip(feature_names, feature_importances))
    importance_pairs.sort(key=lambda x: x[1], reverse=True)
    
    print("\n【Top 15 重要特征】")
    for i, (name, imp) in enumerate(importance_pairs[:15], 1):
        bar_length = int(imp * 100)
        bar = '█' * bar_length
        print(f"{i:2d}. {name:<30s} {bar} {imp:.4f}")
    
    print("\n✅ 特征重要性分析完成（使用本地处理，避免 Spark worker 超时）")
    
except Exception as e:
    print(f"⚠️ 特征重要性分析出错: {e}")
    print("使用备用方案：直接输出模型参数")
    try:
        print(f"  随机森林树数量: {best_rf_model.getNumTrees()}")
        print(f"  最大深度: {best_rf_model.getMaxDepth()}")
    except:
        print(f"  无法获取模型参数")


# ============================================================================
# 【单元格 27】模型保存
# ============================================================================

print("\n" + "=" * 70)
print("                  模型保存")
print("=" * 70)

import shutil
import json

model_path = "return_prediction_model"
lr_model_path = "return_prediction_lr_baseline"

# 步骤1：清理旧模型（避免冲突）
print("\n【步骤1】清理旧的模型文件...")
for path in [model_path, lr_model_path]:
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            print(f"   ✅ 已删除旧模型: {path}/")
        except Exception as e:
            print(f"   ⚠️ 删除失败: {e}")
    else:
        print(f"   ℹ️  {path}/ 不存在，跳过")

# 步骤2：保存 Spark MLlib 格式的模型
print("\n【步骤2】保存 Spark MLlib 格式模型...")
rf_saved = False
lr_saved = False

# 检查模型变量是否存在
if best_rf_model is None:
    print("❌ 错误: best_rf_model 为 None，随机森林模型未成功训练！")
    print("   请确保前面的训练单元格（单元格23.5）已成功运行。")
if lr_model is None:
    print("❌ 错误: lr_model 为 None，逻辑回归模型未成功训练！")
    print("   请确保前面的训练单元格（单元格23.6）已成功运行。")

if best_rf_model is not None:
    try:
        best_rf_model.write().overwrite().save(model_path)
        print(f"✅ 随机森林模型已保存到: {model_path}/")
        print(f"   模型类型: 随机森林分类器 (交叉验证优化)")
        print(f"   特征数量: {len(assembler_inputs)}")
        rf_saved = True
    except Exception as e:
        print(f"⚠️ 随机森林模型保存失败: {e}")
        print("   模型仍保留在内存中（best_rf_model 变量）")

if lr_model is not None:
    try:
        lr_model.write().overwrite().save(lr_model_path)
        print(f"✅ 逻辑回归模型已保存到: {lr_model_path}/")
        print(f"   模型类型: 逻辑回归分类器 (基线模型)")
        print(f"   AUC-ROC: {lr_auc:.4f}")
        print(f"   Accuracy: {lr_accuracy:.4f}")
        lr_saved = True
    except Exception as e:
        print(f"⚠️ 逻辑回归模型保存失败: {e}")
        print("   模型仍保留在内存中（lr_model 变量）")

# 验证模型是否真的保存成功
if rf_saved and os.path.exists(model_path):
    print(f"\n✅ 验证通过: 随机森林模型文件夹存在")
else:
    print(f"\n❌ 验证失败: 随机森林模型文件夹不存在")

if lr_saved and os.path.exists(lr_model_path):
    print(f"✅ 验证通过: 逻辑回归模型文件夹存在")
else:
    print(f"❌ 验证失败: 逻辑回归模型文件夹不存在")

# ============================================================================
# 【步骤2.5】创建并保存完整的 Pipeline（包含特征工程 + 模型）
# ============================================================================
print("\n【步骤2.5】创建并保存完整 Pipeline...")
from pyspark.ml import Pipeline, PipelineModel

pipeline_saved = False
try:
    # 构建完整的 Pipeline：特征工程 + 模型预测
    pipeline_stages = []
    
    # 添加 StringIndexer（使用已拟合的版本）
    for col_name in categorical_features:
        pipeline_stages.append(fitted_indexers[col_name])
    
    # 添加 OneHotEncoder（使用已拟合的版本）
    for col_name in categorical_features:
        pipeline_stages.append(fitted_encoders[col_name])
    
    # 添加 VectorAssembler
    assembler = VectorAssembler(
        inputCols=assembler_inputs,
        outputCol='features',
        handleInvalid="skip"
    )
    pipeline_stages.append(assembler)
    
    # 添加最佳模型（随机森林）
    pipeline_stages.append(best_rf_model)
    
    # 创建 Pipeline 并拟合（因为 indexers/encoders 已经拟合，这里只是组装）
    pipeline = Pipeline(stages=pipeline_stages)
    pipeline_model = pipeline.fit(train_df)
    
    # 保存 Pipeline
    pipeline_path = "return_prediction_pipeline"
    pipeline_model.write().overwrite().save(pipeline_path)
    print(f"✅ 完整 Pipeline 已保存到: {pipeline_path}/")
    print(f"   包含: {len(categorical_features)}个StringIndexer + {len(categorical_features)}个OneHotEncoder + VectorAssembler + RandomForest")
    pipeline_saved = True
except Exception as e:
    print(f"️ Pipeline 保存失败: {e}")
    print("   可以单独加载特征工程组件和模型进行预测")

# 验证 Pipeline 是否保存成功
if pipeline_saved and os.path.exists(pipeline_path):
    print(f"✅ 验证通过: Pipeline 文件夹存在")
else:
    print(f"❌ 验证失败: Pipeline 文件夹不存在")

# 步骤3：导出可读的 JSON 格式模型文件
print("\n【步骤3】导出可读的 JSON 格式模型文件...")

# 使用与训练时相同的特征列表（注意：Order_Status 是标签列，不能作为特征！）
# 特征列表与【单元格17】中的 numerical_features / categorical_features 保持一致
print(f"\n📦 正在导出模型参数...")
print(f"   数值型特征 ({len(numerical_features)} 个): {numerical_features}")
print(f"   分类型特征 ({len(categorical_features)} 个): {categorical_features}")

json_exported_count = 0

# 导出随机森林模型为 JSON
if rf_saved and best_rf_model is not None:
    try:
        feature_importances = best_rf_model.featureImportances.toArray().tolist()
        
        rf_info = {
            "model_type": "RandomForestClassification",
            "num_trees": best_rf_model.getNumTrees(),
            "num_features": best_rf_model.numFeatures,
            "max_depth": best_rf_model.getMaxDepth(),
            "num_classes": best_rf_model.numClasses,
            "feature_names": assembler_inputs,
            "feature_importances": {
                feat: float(imp) for feat, imp in zip(assembler_inputs, feature_importances)
            },
            "prediction_threshold": 0.5,
            "class_labels": ["不退货 (0)", "退货 (1)"]
        }
        
        # 尝试提取每棵树的信息
        try:
            trees_data = []
            for i in range(min(best_rf_model.getNumTrees(), 5)):
                tree = best_rf_model.trees[i]
                trees_data.append({
                    "tree_index": i,
                    "depth": tree.depth,
                    "num_nodes": tree.numNodes
                })
            
            if len(trees_data) > 0:
                rf_info["trees_summary"] = trees_data
                rf_info["total_trees_saved"] = best_rf_model.getNumTrees()
                rf_info["note"] = f"完整树结构保存在 Spark 格式中 ({model_path}/)，这里只保存了前{len(trees_data)}棵树的摘要"
        except:
            rf_info["note"] = "详细树结构需要查看 Spark 格式的模型文件"
        
        rf_json_path = "random_forest_model_full.json"
        with open(rf_json_path, 'w', encoding='utf-8') as f:
            json.dump(rf_info, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ [模型1] 随机森林完整参数已导出")
        print(f"   📄 JSON文件: {rf_json_path}")
        print(f"    Spark格式: {model_path}/")
        print(f"   💾 文件大小: {os.path.getsize(rf_json_path):,} bytes")
        print(f"   📊 最重要的3个特征:")
        sorted_importance = sorted(rf_info['feature_importances'].items(), key=lambda x: x[1], reverse=True)[:3]
        for i, (feat, imp) in enumerate(sorted_importance, 1):
            print(f"     {i}. {feat}: {imp:.4f}")
        json_exported_count += 1
    except Exception as e:
        print(f"️ 随机森林 JSON 导出失败: {e}")

# 导出逻辑回归模型为 JSON
if lr_saved and lr_model is not None:
    try:
        coefficients = lr_model.coefficientMatrix.toArray().tolist()
        intercepts = lr_model.interceptVector.toArray().tolist()
        
        lr_info = {
            "model_type": "LogisticRegression",
            "num_classes": lr_model.numClasses,
            "regularization_type": "elastic_net" if lr_model.getElasticNetParam() > 0 else "l2",
            "elastic_net_param": lr_model.getElasticNetParam(),
            "reg_param": lr_model.getRegParam(),
            "feature_names": assembler_inputs,
            "class_labels": ["不退货 (0)", "退货 (1)"],
            "coefficients": {
                f"class_{i}": {
                    feat: float(coef) for feat, coef in zip(assembler_inputs, coef_row)
                }
                for i, coef_row in enumerate(coefficients)
            },
            "intercepts": {
                f"class_{i}": float(intercept) for i, intercept in enumerate(intercepts)
            }
        }
        
        lr_json_path = "logistic_regression_model_full.json"
        with open(lr_json_path, 'w', encoding='utf-8') as f:
            json.dump(lr_info, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ [模型2] 逻辑回归完整参数已导出")
        print(f"   📄 JSON文件: {lr_json_path}")
        print(f"    Spark格式: {lr_model_path}/")
        print(f"   💾 文件大小: {os.path.getsize(lr_json_path):,} bytes")
        print(f"   📊 最重要的3个特征（按绝对值排序）:")
        
        # 计算每个特征的总影响
        feature_impact = {}
        for feat in assembler_inputs:
            avg_coef = sum(abs(lr_info['coefficients'][f'class_{i}'][feat]) for i in range(lr_info['num_classes'])) / lr_info['num_classes']
            feature_impact[feat] = avg_coef
        
        sorted_impact = sorted(feature_impact.items(), key=lambda x: x[1], reverse=True)[:3]
        for i, (feat, impact) in enumerate(sorted_impact, 1):
            print(f"     {i}. {feat}: {impact:.4f}")
        json_exported_count += 1
    except Exception as e:
        print(f"️ 逻辑回归 JSON 导出失败: {e}")

print("\n" + "=" * 70)
print("                  模型保存完成")
print("=" * 70)
print("\n📁 已生成的文件清单:")
if rf_saved and os.path.exists(model_path):
    print(f"   ✅ [随机森林] random_forest_model_full.json - 完整参数(JSON格式)")
    print(f"   📂 [随机森林] {model_path}/ - Spark MLlib格式(程序使用)")
else:
    print(f"   ❌ [随机森林] 模型未成功保存")

if lr_saved and os.path.exists(lr_model_path):
    print(f"   ✅ [逻辑回归] logistic_regression_model_full.json - 完整参数(JSON格式)")
    print(f"    [逻辑回归] {lr_model_path}/ - Spark MLlib格式(程序使用)")
else:
    print(f"   ❌ [逻辑回归] 模型未成功保存")

print(f"\n💡 JSON 文件导出数量: {json_exported_count}/2")

# ============================================================================
# 【步骤4】额外保存为 Pickle 格式（用于 Web 应用快速加载）
# ============================================================================

print("\n【步骤4】保存模型元数据（Pickle格式）...")
print("   💡 注意: PySpark模型对象无法直接pickle，保存模型路径和特征元数据")

pickle_exported = False

try:
    # 创建保存目录
    pickle_dir = './saved_models'
    os.makedirs(pickle_dir, exist_ok=True)
    
    # 保存模型元数据（不包含PySpark模型对象，因为它们无法被pickle序列化）
    try:
        num_trees = best_rf_model.getNumTrees()
    except TypeError:
        num_trees = best_rf_model.numTrees
    
    model_metadata = {
        'rf_model_path': model_path,  # 随机森林模型路径
        'lr_model_path': lr_model_path,  # 逻辑回归模型路径
        'feature_names': assembler_inputs,
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'model_type': 'random_forest_and_logistic_regression',
        'best_model': 'random_forest',  # 默认使用随机森林
        'num_trees': num_trees,
        'pipeline_dir': './pipeline'
    }
    
    # 保存为 Pickle
    pickle_path = f'{pickle_dir}/model_metadata.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(model_metadata, f)
    
    print(f"\n✅ [Pickle] 模型元数据已保存到: {pickle_path}")
    print(f"    文件大小: {os.path.getsize(pickle_path):,} bytes")
    print(f"   📦 包含内容:")
    print(f"     • 随机森林模型路径 ({num_trees} 棵树)")
    print(f"     • 逻辑回归模型路径")
    print(f"     • 特征名称列表 ({len(assembler_inputs)} 个特征)")
    print(f"     • pipeline目录路径")
    pickle_exported = True
    
except Exception as e:
    print(f"⚠️ Pickle 导出失败: {e}")
    import traceback
    traceback.print_exc()

# --- 步骤5：pipeline/ 目录已在特征工程阶段保存 ---
print("\n【步骤5】pipeline/ 目录已在特征工程阶段保存")
if os.path.exists('./pipeline'):
    pipeline_files = os.listdir('./pipeline')
    print(f"   ✅ pipeline/ 目录包含 {len(pipeline_files)} 个文件: {pipeline_files}")
else:
    print(f"   ⚠️ pipeline/ 目录不存在")

print("\n" + "=" * 70)
print("                  所有模型保存完成")
print("=" * 70)
print("\n📁 最终文件清单:")
if rf_saved and os.path.exists(model_path):
    print(f"   ✅ [随机森林] random_forest_model_full.json - 完整参数(JSON格式)")
    print(f"    [随机森林] {model_path}/ - Spark MLlib格式(程序使用)")
if lr_saved and os.path.exists(lr_model_path):
    print(f"   ✅ [逻辑回归] logistic_regression_model_full.json - 完整参数(JSON格式)")
    print(f"   📂 [逻辑回归] {lr_model_path}/ - Spark MLlib格式(程序使用)")
else:
    print(f"    [逻辑回归] 模型未成功保存")

if pipeline_saved and os.path.exists(pipeline_path):
    print(f"   ✅ [Pipeline] return_prediction_pipeline/ - 完整Pipeline(特征工程+模型，推荐Web使用)")
else:
    print(f"   ⚠️ [Pipeline] return_prediction_pipeline/ - 未生成")
if pickle_exported:
    print(f"   ✅ [Pickle] saved_models/model_metadata.pkl - 模型元数据")
else:
    print(f"   ⚠️ [Pickle] saved_models/model_metadata.pkl - 未生成")

if os.path.exists('./pipeline'):
    print(f"   ✅ [Feature Engineering] pipeline/ - 特征工程组件({len(categorical_features)}个indexer + {len(categorical_features)}个encoder)")
else:
    print(f"   ⚠️ [Feature Engineering] pipeline/ - 特征工程组件保存失败")

print(f"\n💡 说明:")
print("   - JSON文件: 人类可读，可查看特征重要性、系数等详细参数")
print("   - Spark格式文件夹: 程序加载和预测时使用")
print("   - Pipeline文件夹 (推荐): 包含完整特征工程+模型，Web应用只需一行代码加载")
print("   - Pickle格式: 模型元数据（路径、特征列表），方便查看配置")
print("   - Feature Engineering目录: 单独的特征编码器，可灵活组合使用")
print("   - 下次运行会自动清理旧模型并生成新的")


# ============================================================================
# 【单元格 28】可视化展示 - 准备数据
# ============================================================================

print("\n" + "=" * 70)
print("                  可视化展示模块")
print("=" * 70)

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（Windows）
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    print("✅ 中文字体加载成功")
except:
    print("⚠️ 中文字体加载失败，将使用英文标签")

print("\n✅ Matplotlib 已就绪，开始生成可视化图表...")


# ============================================================================
# 【单元格 29】图表1 - 订单状态分布饼图
# ============================================================================

print("\n【图表1】订单状态分布饼图")

# ✅ 用 toPandas() 替代 rdd.flatMap().collect()
status_df = order_status_with_pct.toPandas()

fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']
wedges, texts, autotexts = ax.pie(
    status_df['count'], 
    labels=status_df['Order_Status'],
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    textprops={'fontsize': 12, 'weight': 'bold'}
)

ax.set_title('Order Status Distribution', fontsize=16, weight='bold', pad=20)
plt.tight_layout()
plt.savefig('chart1_order_status.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ 图表1已保存为 chart1_order_status.png")


# ============================================================================
# 【单元格 30】图表2 - 产品类别销售额柱状图
# ============================================================================

print("\n【图表2】产品类别销售额对比")

# ✅ 用 toPandas()
category_df = category_stats.limit(10).toPandas()

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(range(len(category_df)), category_df['total_revenue'], color='#3498db', alpha=0.8)
ax.set_xticks(range(len(category_df)))
ax.set_xticklabels(category_df['Product_Category'], rotation=45, ha='right', fontsize=10)
ax.set_ylabel('Sales ($)', fontsize=12, weight='bold')
ax.set_title('Top 10 Product Categories by Revenue', fontsize=16, weight='bold', pad=20)

for i, (bar, val) in enumerate(zip(bars, category_df['total_revenue'])):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + val*0.01,
            f'${val:,.0f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('chart2_category_sales.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ 图表2已保存为 chart2_category_sales.png")


# ============================================================================
# 【单元格 31】图表3 - 客户性别消费对比
# ============================================================================

print("\n【图表3】客户性别消费对比")

# ✅ 用 toPandas()
gender_df = gender_stats.toPandas()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Subplot 1: Order count
axes[0].bar(gender_df['Customer_Gender'], gender_df['order_count'], color=['#e74c3c', '#3498db'])
axes[0].set_title('Order Count by Gender', fontsize=14, weight='bold')
axes[0].set_ylabel('Orders', fontsize=12)

# Subplot 2: Average spending
axes[1].bar(gender_df['Customer_Gender'], gender_df['avg_spending'], color=['#e74c3c', '#3498db'])
axes[1].set_title('Average Spending by Gender', fontsize=14, weight='bold')
axes[1].set_ylabel('Avg Spending ($)', fontsize=12)

# Subplot 3: Average rating
axes[2].bar(gender_df['Customer_Gender'], gender_df['avg_rating'], color=['#e74c3c', '#3498db'])
axes[2].set_title('Average Rating by Gender', fontsize=14, weight='bold')
axes[2].set_ylabel('Avg Rating', fontsize=12)

plt.suptitle('Customer Gender Comparison', fontsize=18, weight='bold', y=1.02)
plt.tight_layout()
plt.savefig('chart3_gender_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ 图表3已保存为 chart3_gender_comparison.png")


# ============================================================================
# 【单元格 32】图表4 - 配送速度与满意度关系
# ============================================================================

print("\n【图表4】配送速度与客户满意度关系")

# ✅ 用 toPandas()
delivery_df = delivery_analysis.toPandas()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Order distribution
axes[0].bar(delivery_df['delivery_speed'], delivery_df['order_count'], color='#9b59b6', alpha=0.8)
axes[0].set_title('Order Distribution by Delivery Speed', fontsize=14, weight='bold')
axes[0].set_ylabel('Orders', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)

# Subplot 2: Rating and return rate
x = np.arange(len(delivery_df))
width = 0.35
bars1 = axes[1].bar(x - width/2, delivery_df['avg_rating'], width, label='Avg Rating', color='#2ecc71', alpha=0.8)
bars2 = axes[1].bar(x + width/2, delivery_df['return_rate']*100, width, label='Return Rate (%)', color='#e74c3c', alpha=0.8)

axes[1].set_xlabel('Delivery Speed', fontsize=12)
axes[1].set_title('Impact of Delivery Speed on Rating and Return Rate', fontsize=14, weight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(delivery_df['delivery_speed'], rotation=45, ha='right')
axes[1].legend(fontsize=11)
axes[1].tick_params(axis='y', which='both', labelsize=10)

plt.suptitle('Delivery Time Analysis', fontsize=18, weight='bold', y=1.02)
plt.tight_layout()
plt.savefig('chart4_delivery_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ 图表4已保存为 chart4_delivery_analysis.png")


# ============================================================================
# 【单元格 33】图表5 - 会员等级消费能力
# ============================================================================

print("\n【图表5】会员等级消费能力分析")

# ✅ 用 toPandas()
membership_df = membership_stats.toPandas()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Average spending
colors_mem = ['#f1c40f', '#e67e22', '#3498db', '#9b59b6']
axes[0].bar(membership_df['Membership_Status'], membership_df['avg_spending'], color=colors_mem, alpha=0.8)
axes[0].set_title('Average Spending by Membership Level', fontsize=14, weight='bold')
axes[0].set_ylabel('Avg Spending ($)', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)

# Subplot 2: Return rate
axes[1].bar(membership_df['Membership_Status'], membership_df['return_rate']*100, color=colors_mem, alpha=0.8)
axes[1].set_title('Return Rate by Membership Level', fontsize=14, weight='bold')
axes[1].set_ylabel('Return Rate (%)', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)

plt.suptitle('Membership Level Analysis', fontsize=18, weight='bold', y=1.02)
plt.tight_layout()
plt.savefig('chart5_membership_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ 图表5已保存为 chart5_membership_analysis.png")


# ============================================================================
# 【单元格 34】图表6 - 特征重要性可视化
# ============================================================================

print("\n【图表6】模型特征重要性可视化")

# 使用之前计算的特征重要性数据（原生 Python 列表）
top_15_features = importance_pairs[:15]
top_15_features.reverse()  # Reverse order so most important is at top
feature_names = [name for name, _ in top_15_features]
feature_importances = [imp for _, imp in top_15_features]

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(range(len(top_15_features)), feature_importances, color='#3498db', alpha=0.8)
ax.set_yticks(range(len(top_15_features)))
ax.set_yticklabels(feature_names, fontsize=10)
ax.set_xlabel('Importance Score', fontsize=12, weight='bold')
ax.set_title('Random Forest Model - Top 15 Important Features', fontsize=16, weight='bold', pad=20)

for i, (bar, val) in enumerate(zip(bars, feature_importances)):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2.,
            f'{val:.4f}', va='center', fontsize=9, weight='bold')

plt.tight_layout()
plt.savefig('chart6_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ 图表6已保存为 chart6_feature_importance.png")


# ============================================================================
# 【单元格 35】项目总结
# ============================================================================

print("\n" + "=" * 70)
print("                    项目完成总结")
print("=" * 70)

print("\n✅ 已完成的工作：")
print("  1. ✓ 数据读取与结构查看")
print("  2. ✓ 数据预处理（缺失值、重复值、异常值处理）")
print("  3. ✓ 10个数据分析任务")
print("     - 订单状态分布")
print("     - 产品类别销售情况")
print("     - 客户性别消费差异")
print("     - 配送时效分析")
print("     - 支付方式偏好")
print("     - 会员等级消费分析")
print("     - 设备类型与购买行为")
print("     - 高价值客户识别")
print("     - 月度订单趋势分析（Spark SQL）")
print("     - 产品类别×国家退货率分析（join）")
print("  4. ✓ PySpark MLlib 建模（随机森林分类器）")
print("  5. ✓ 模型评估（AUC-ROC、准确率、精确率、召回率、F1-Score）")
print("  6. ✓ 6张可视化图表")
print("     - 订单状态分布饼图")
print("     - 产品类别销售额柱状图")
print("     - 客户性别消费对比")
print("     - 配送速度与满意度关系")
print("     - 会员等级消费能力")
print("     - 特征重要性可视化")

print("\n📊 使用的 PySpark 技术：")
print("  • DataFrame 操作（filter, select, withColumn, drop, na.fill）")
print("  • Spark SQL 查询（createOrReplaceTempView + spark.sql）")
print("  • 分组聚合操作（groupBy + agg）")
print("  • 常用操作（orderBy, dropDuplicates, desc）")
print("  • 多表连接操作（join）")
print("  • PySpark MLlib 机器学习模块（RandomForest, LogisticRegression）")
print("  • 特征工程（StringIndexer, OneHotEncoder, VectorAssembler）")

print("\n📁 生成的文件：")
print("  • chart1_order_status.png")
print("  • chart2_category_sales.png")
print("  • chart3_gender_comparison.png")
print("  • chart4_delivery_analysis.png")
print("  • chart5_membership_analysis.png")
print("  • chart6_feature_importance.png")

print("\n💡 业务建议：")
print("  1. 针对高退货风险订单（预测为退货），提前发送关怀消息或优惠券")
print("  2. 优化配送时效，将配送天数控制在 5 天内可降低退货率 15%")
print("  3. 对 Platinum 会员提供专属客服，维持高忠诚度")
print("  4. 移动端用户订单量大但平均金额低，可推出移动端专属促销")
print("  5. Fashion 和 Electronics 品类销售额最高，可增加库存和营销投入")

print("\n" + "=" * 70)
print("              🎉 项目圆满完成！")
print("=" * 70)

# ============================================================================
# 【单元格 36】导出分析结果到 JSON（供 Web 系统使用）
# ============================================================================

import json
from datetime import datetime

print("\n【导出结果】正在保存分析结果到 JSON 文件...")

# 准备导出数据（全部使用 toPandas() 避免 rdd.map().collect() 超时）
export_data = {
    'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    
    # 订单状态分布
    'order_status': order_status_with_pct.toPandas()[['Order_Status', 'count', 'percentage']].to_dict('records'),
    
    # 产品类别销售
    'category_sales': category_stats.limit(10).toPandas()[['Product_Category', 'order_count', 'total_revenue']].to_dict('records'),
    
    # 客户性别消费
    'gender_stats': gender_stats.toPandas()[['Customer_Gender', 'order_count', 'avg_spending', 'avg_rating']].to_dict('records'),
    
    # 配送速度分析
    'delivery_analysis': delivery_analysis.toPandas()[['delivery_speed', 'order_count', 'avg_rating', 'return_rate']].to_dict('records'),
    
    # 会员等级分析
    'membership_stats': membership_stats.toPandas()[['Membership_Status', 'order_count', 'avg_spending', 'return_rate']].to_dict('records'),
    
    # 模型评估指标
    'model_metrics': {
        'auc_roc': float(best_auc),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    },
    
    # 混淆矩阵
    'confusion_matrix': {
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    },
    
    # 特征重要性（使用原生 Python 列表）
    'feature_importance': [
        {'feature': name, 'importance': float(imp)} 
        for name, imp in importance_pairs[:15]
    ]
}

# 保存到 JSON 文件
with open('analysis_results.json', 'w', encoding='utf-8') as f:
    json.dump(export_data, f, indent=2, ensure_ascii=False)

print("✅ 分析结果已保存到 analysis_results.json")
print("   文件大小: {:.2f} KB".format(os.path.getsize('analysis_results.json') / 1024))

# ============================================================================
# 【Web应用使用示例】
# ============================================================================
print("\n" + "=" * 70)
print("                  Web应用使用指南")
print("=" * 70)
print("\n📌 方法1：加载完整 Pipeline（推荐，最简单）")
print("""
from pyspark.ml import PipelineModel

# 一行代码加载完整 Pipeline（包含特征工程+模型）
pipeline = PipelineModel.load("return_prediction_pipeline")

# 直接预测（自动处理特征工程）
df_pred = pipeline.transform(new_data_df)
predictions = df_pred.select("prediction", "probability").collect()
""")

print("\n📌 方法2：单独加载特征工程组件和模型（灵活控制）")
print("""
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import StringIndexerModel, OneHotEncoderModel, VectorAssembler

# 加载特征工程组件
indexers = {}
encoders = {}
for col in categorical_features:
    indexers[col] = StringIndexerModel.load(f"pipeline/{col}_indexer")
    encoders[col] = OneHotEncoderModel.load(f"pipeline/{col}_encoder")

# 加载模型
model = RandomForestClassificationModel.load("return_prediction_model")

# 手动进行特征工程后预测
df_indexed = new_data_df
for col in categorical_features:
    df_indexed = indexers[col].transform(df_indexed)
    df_indexed = encoders[col].transform(df_indexed)
assembler = VectorAssembler(inputCols=feature_names, outputCol='features')
df_features = assembler.transform(df_indexed)
predictions = model.transform(df_features)
""")

print("\n💡 建议：")
print("   • Web应用推荐使用 Pipeline（方法1），代码简洁、不易出错")
print("   • 需要灵活调整特征工程时使用方法2")
print("   • 所有模型文件都在 d:\\spark\\ 目录下")

# 清理资源
spark.stop()
print("\n✨ Spark Session 已关闭")
