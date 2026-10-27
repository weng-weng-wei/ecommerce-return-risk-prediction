#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试独立预测服务是否正常工作
使用方法：python test_predict_service.py
"""

import requests
import json

PREDICT_SERVICE_URL = 'http://localhost:5001/predict'
HEALTH_CHECK_URL = 'http://localhost:5001/health'

def test_health():
    """测试健康检查接口"""
    print("\n" + "="*70)
    print(" 测试健康检查接口")
    print("="*70)
    
    try:
        response = requests.get(HEALTH_CHECK_URL, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 健康检查成功")
            print(f"   状态: {data.get('status')}")
            print(f"   模型加载: {data.get('model_loaded')}")
            print(f"   Spark就绪: {data.get('spark_ready')}")
            return True
        else:
            print(f"❌ 健康检查失败，状态码: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到预测服务")
        print("💡 请先启动预测服务: python predict_service.py")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_prediction():
    """测试预测接口"""
    print("\n" + "="*70)
    print("🔮 测试预测接口")
    print("="*70)
    
    # 构建测试数据（使用默认值）
    test_data = {
        'customer_age': 42,
        'unit_price': 200.0,
        'quantity': 2,
        'discount_percent': 15.0,
        'shipping_cost': 15.0,
        'tax_amount': 12.0,
        'order_amount': 180.0,
        'delivery_days': 6,
        'review_rating': 3.8,
        'customer_lifetime_value': 500.0,
        'profit_margin_percent': 25.0,
        'profit_amount': 45.0,
        'customer_gender': 'Female',
        'country': 'United States',
        'customer_segment': 'Regular',
        'product_category': 'Fashion',
        'payment_method': 'Credit Card',
        'device_type': 'Mobile',
        'traffic_source': 'Website',
        'membership_status': 'Silver',
        'shipping_method': 'Standard',
        'season': 'Winter',
        'holiday_season': 'No'
    }
    
    try:
        response = requests.post(
            PREDICT_SERVICE_URL,
            json=test_data,
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"✅ 预测成功")
                print(f"   预测结果: {'退货' if result['prediction'] == 1 else '不退货'}")
                print(f"   退货概率: {result['probability']:.2%}")
                print(f"   风险等级: {result['risk_level']}")
                print(f"   模型类型: {result['model_type']}")
                print(f"   消息: {result['message']}")
                return True
            else:
                print(f"❌ 预测失败: {result.get('error')}")
                return False
        else:
            print(f"❌ 预测请求失败，状态码: {response.status_code}")
            print(f"   响应: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到预测服务")
        print("💡 请先启动预测服务: python predict_service.py")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_high_risk_scenario():
    """测试高风险场景"""
    print("\n" + "="*70)
    print("️  测试高风险场景")
    print("="*70)
    
    # 构建高风险数据
    high_risk_data = {
        'customer_age': 25,
        'unit_price': 500.0,
        'quantity': 1,
        'discount_percent': 45.0,  # 高折扣
        'shipping_cost': 30.0,
        'tax_amount': 20.0,
        'order_amount': 450.0,
        'delivery_days': 18,  # 超长配送时间
        'review_rating': 2.0,  # 低评分
        'customer_lifetime_value': 100.0,
        'profit_margin_percent': 10.0,
        'profit_amount': 45.0,
        'customer_gender': 'Male',
        'country': 'China',
        'customer_segment': 'New',
        'product_category': 'Fashion',  # 高退货率类别
        'payment_method': 'Cash on Delivery',  # 货到付款
        'device_type': 'Desktop',
        'traffic_source': 'Advertisement',
        'membership_status': 'Bronze',
        'shipping_method': 'Economy',
        'season': 'Summer',
        'holiday_season': 'Yes'
    }
    
    try:
        response = requests.post(
            PREDICT_SERVICE_URL,
            json=high_risk_data,
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"✅ 高风险场景预测成功")
                print(f"   预测结果: {'退货' if result['prediction'] == 1 else '不退货'}")
                print(f"   退货概率: {result['probability']:.2%}")
                print(f"   风险等级: {result['risk_level']}")
                
                # 验证是否为高风险
                if result['probability'] > 0.5:
                    print(f"✅ 正确识别为高风险订单")
                else:
                    print(f"⚠️  未识别为高风险（概率: {result['probability']:.2%}）")
                
                return True
            else:
                print(f"❌ 预测失败: {result.get('error')}")
                return False
        else:
            print(f"❌ 预测请求失败，状态码: {response.status_code}")
            return False
            
    except Exception as e:
        print(f" 测试失败: {e}")
        return False

if __name__ == '__main__':
    print("\n" + "="*70)
    print("         🔧 独立预测服务测试工具")
    print("="*70)
    
    results = []
    
    # 运行测试
    results.append(("健康检查", test_health()))
    results.append(("基础预测", test_prediction()))
    results.append(("高风险场景", test_high_risk_scenario()))
    
    # 汇总结果
    print("\n" + "="*70)
    print("📊 测试结果汇总")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {name:20s} {status}")
    
    print("-"*70)
    print(f"   总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！预测服务工作正常")
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查日志")
    
    print("="*70)
