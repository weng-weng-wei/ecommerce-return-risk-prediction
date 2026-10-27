# 🚀 快速开始 - 智能退货预测系统

## ⚡ 3步启动（推荐）

### 1️⃣ 运行启动脚本

```powershell
.\start_services.ps1
```

这个脚本会自动：
- ✅ 检查 Python 环境和依赖
- ✅ 检查模型文件
- ✅ 启动预测服务（端口 5001）
- ✅ 启动 Web Dashboard（端口 5000）

### 2️⃣ 打开浏览器

访问：**http://localhost:5000**

### 3️⃣ 开始使用

- 在"实时预测"页面填写订单信息
- 点击"预测"按钮查看结果
- 查看风险等级和退货概率

---

## 🔍 手动启动（可选）

如果启动脚本有问题，可以手动启动：

### 终端1：启动预测服务

```powershell
python predict_service.py
```

等待看到：`🎉 模型加载成功，服务就绪！`

### 终端2：启动 Web Dashboard

```powershell
python web_dashboard.py
```

等待看到：`访问地址: http://localhost:5000`

---

## 🧪 测试验证

运行测试脚本验证服务是否正常：

```powershell
python test_predict_service.py
```

预期输出：
```
✅ 健康检查成功
✅ 基础预测成功
✅ 高风险场景预测成功
 所有测试通过！
```

---

## ❓ 常见问题

### Q1: 启动时提示"未找到 Python"？

**A:** 确保 Python 已安装并添加到 PATH：
```powershell
python --version
```

### Q2: 提示缺少依赖包？

**A:** 安装所需依赖：
```powershell
pip install flask pyspark requests flask-cors
```

### Q3: 预测服务启动失败？

**A:** 
1. 检查是否有其他程序占用 5001 端口
2. 查看终端错误信息
3. 尝试重启计算机

### Q4: Web Dashboard 显示"规则引擎模式"？

**A:** 这是正常的！Web Dashboard 默认使用规则引擎作为备用方案。如果要使用真实模型：
1. 确保预测服务已启动（访问 http://localhost:5001/health）
2. 修改 `web_dashboard.py` 中的 `FORCE_RULE_ENGINE = False`

### Q5: 如何停止服务？

**A:** 
- **方法1**: 关闭两个 Python 窗口
- **方法2**: 在任务管理器中结束 python.exe 进程
- **方法3**: 按 Ctrl+C（在某些情况下可能无效）

---

## 📊 系统架构

```
┌─────────────────┐         ┌──────────────────────┐
│  Web Dashboard   │         │  独立预测服务         │
│  (端口 5000)     │  HTTP   │  (端口 5001)         │
│                  │──────▶  │                      │
│  • 可视化界面    │         │  • PySpark Pipeline  │
│  • 预测 API      │         │  • 特征工程          │
│  • 数据看板      │         │  • 模型推理          │
└─────────────────┘         └──────────────────────┘
```

---

## 📁 项目文件说明

| 文件 | 说明 |
|------|------|
| `predict_service.py` | 独立预测服务（核心） |
| `web_dashboard.py` | Web 可视化界面 |
| `start_services.ps1` | 一键启动脚本 |
| `test_predict_service.py` | 测试工具 |
| `ecommerce_complete_project.py` | 训练脚本 |
| `return_prediction_pipeline/` | 训练好的 Pipeline 模型 |
| `pipeline/` | 特征编码器 |
| `README_预测服务.md` | 详细文档 |
| `QUICKSTART.md` | 本文件 |

---

## 🎯 下一步

- 📖 查看详细文档：[README_预测服务.md](README_预测服务.md)
- 🧪 运行完整测试：`python test_predict_service.py`
- 📊 查看训练数据：先运行 `python ecommerce_complete_project.py`

---

## 💡 提示

- 首次启动可能需要 1-2 分钟加载模型
- 预测服务使用约 2-3GB 内存
- 建议至少 8GB RAM 以获得最佳性能
- Windows 用户建议使用 PowerShell 而非 CMD

---

**祝您使用愉快！** 🎉

如有问题，请查看终端日志或提交 Issue。
# 🚀 快速启动指南 - PySpark 电商数据分析 Web 系统

## ⚡ 两步启动（推荐）

### Step 1: 生成真实数据
```bash
cd d:\spark
python ecommerce_complete_project.py
```
✅ 这会生成 `analysis_results.json` 文件

### Step 2: 启动 Web 系统
**方法一：** 双击 `start_web.bat`

**方法二：** 命令行
```bash
cd d:\spark
python web_dashboard.py
```

🌐 **访问地址：** http://localhost:5000

---

## 📋 详细说明

### 前置条件
- ✅ Python 3.x 已安装
- ✅ Flask 已安装（如未安装：`pip install flask`）
- ✅ PySpark 环境已配置

### 运行顺序（重要！）

#### 1️ 先运行 PySpark 分析
```bash
python ecommerce_complete_project.py
```

**耗时：** 5-10 分钟  
**输出：** 
- `analysis_results.json` ← Web 系统需要的数据
- `chart1-6.png` ← 可视化图表
- 控制台输出所有分析结果

#### 2️⃣ 再启动 Web 系统
```bash
python web_dashboard.py
```

**状态提示：**
- ✅ 如果看到 "检测到真实数据" → 使用 PySpark 生成的真实数据
- ⚠️ 如果看到 "使用模拟数据" → 请先运行 Step 1

---

## 🔍 故障排查

### Q1: Web 系统显示 "undefined" 或 "NaN%"
**原因：** JSON 文件格式问题  
**解决：** 确保已运行 `ecommerce_complete_project.py` 生成最新数据

### Q2: ModuleNotFoundError: No module named 'flask'
**解决：**
```bash
pip install flask
```

### Q3: 端口 5000 被占用
**解决：** 修改 `web_dashboard.py` 第 294 行的 `port=5000` 为其他端口

### Q4: 特征重要性图表不显示
**解决：** 刷新浏览器页面（Ctrl+F5 强制刷新）

---

## 🎯 功能清单

| 页面 | 路径 | 功能 |
|------|------|------|
| 主页仪表盘 | `/` | 统计卡片 + 订单分布饼图 |
| 订单状态分析 | `/analysis/order-status` | 订单分布详情 |
| 产品销售分析 | `/analysis/category-sales` | Top 10 产品类别 |
| 客户行为分析 | `/analysis/customer-behavior` | 性别消费对比 |
| 配送时效分析 | `/analysis/delivery` | 配送速度 vs 评分/退货率 |
| 会员等级分析 | `/analysis/membership` | 不同会员等级消费能力 |
| ML 模型分析 | `/analysis/ml-model` | 模型评估 + 特征重要性 |
| **退货预测** | `/predict` | **实时预测订单退货风险**  |

---

## 💡 使用技巧

1. **查看真实数据：** 打开 `analysis_results.json` 查看原始数据
2. **刷新缓存：** 按 Ctrl+F5 强制刷新浏览器
3. **查看日志：** Web 系统控制台会显示数据加载状态
4. **预测功能：** 在 `/predict` 页面输入订单信息获取退货概率

---

## 📞 技术支持

如有问题，请检查：
1. ✅ 是否已运行 `ecommerce_complete_project.py`
2. ✅ `analysis_results.json` 是否存在于 `d:\spark\` 目录
3. ✅ Flask 是否已安装
4. ✅ 端口 5000 是否可用

---

**祝您使用愉快！** 🎉
