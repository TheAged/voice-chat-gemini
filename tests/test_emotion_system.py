#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三模態情緒辨識系統測試腳本
用於驗證各個模組功能正常運作
"""

import sys
import os

# 加入當前目錄到路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """測試模組導入"""
    print("測試模組導入...")
    
    try:
        import emotion_module
        print("emotion_module 導入成功")
    except Exception as e:
        print(f"emotion_module 導入失敗: {e}")
        return False
    
    try:
        import emotion_config
        print("emotion_config 導入成功")
    except Exception as e:
        print(f"emotion_config 導入失敗: {e}")
        return False
    
    return True

def test_configuration():
    """測試配置系統"""
    print("\n測試配置系統...")
    
    try:
        from emotion_config import get_current_config, print_config_status
        
        config = get_current_config()
        print(f"配置讀取成功: {config['mode']}")
        
        print_config_status()
        return True
        
    except Exception as e:
        print(f"配置系統測試失敗: {e}")
        return False

def test_text_emotion():
    """測試文字情緒辨識"""
    print("\n測試文字情緒辨識...")
    
    try:
        from emotion_module import detect_text_emotion
        
        test_cases = [
            "今天真的很開心！",
            "我覺得有點難過...",
            "這讓我很生氣！",
            "今天天氣不錯。"
        ]
        
        for text in test_cases:
            emotion = detect_text_emotion(text)
            print(f"「{text}」 → {emotion}")
            
        return True
        
    except Exception as e:
        print(f"文字情緒辨識測試失敗: {e}")
        return False

def test_facial_emotion():
    """測試臉部情緒辨識（模擬模式）"""
    print("\n 測試臉部情緒辨識...")
    
    try:
        from emotion_module import detect_facial_emotion, set_facial_recognition_mode
        
        # 確保使用模擬模式
        set_facial_recognition_mode(simulation=True)
        
        # 測試多次以確保隨機性
        print("模擬模式測試（應該看到不同的隨機結果）:")
        for i in range(3):
            emotion, confidence = detect_facial_emotion()
            print(f" 模擬結果 {i+1}: {emotion} (信心度: {confidence})")
            
        return True
        
    except Exception as e:
        print(f"❌ 臉部情緒辨識測試失敗: {e}")
        return False

def test_emotion_fusion():
    """測試情緒融合"""
    print("\n 測試情緒融合...")
    
    try:
        from emotion_module import fuse_emotions
        
        # 測試不同模態組合
        test_cases = [
            {"name": "純文字", "text": "快樂", "audio": None, "facial": None},
            {"name": "文字+語音", "text": "快樂", "audio": "悲傷", "facial": None},
            {"name": "全模態", "text": "快樂", "audio": "悲傷", "facial": "中性"}
        ]
        
        for case in test_cases:
            final_emotion, confidence = fuse_emotions(
                text_emotion=case["text"],
                audio_emotion=case["audio"],
                facial_emotion=case["facial"]
            )
            print(f" {case['name']}: {final_emotion}")
            
        return True
        
    except Exception as e:
        print(f" 情緒融合測試失敗: {e}")
        return False

def test_multi_modal():
    """測試多模態情緒辨識"""
    print("\n 測試多模態情緒辨識...")
    
    try:
        from emotion_module import multi_modal_emotion_detection
        
        # 測試不同組合
        test_cases = [
            {"text": "今天心情不錯！", "audio": None, "facial": False},
            {"text": "感覺有點累...", "audio": "不存在的音檔.wav", "facial": True},
        ]
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n測試案例 {i}:")
            final_emotion, details = multi_modal_emotion_detection(
                text=case["text"],
                audio_path=case["audio"],
                enable_facial=case["facial"]
            )
            print(f" 最終情緒: {final_emotion}")
            print(f" 使用模態: {', '.join(details['modalities_used'])}")
            
        return True
        
    except Exception as e:
        print(f" 多模態情緒辨識測試失敗: {e}")
        return False

def run_all_tests():
    """執行所有測試"""
    print(" 三模態情緒辨識系統測試")
    print("=" * 50)
    
    tests = [
        ("模組導入", test_imports),
        ("配置系統", test_configuration), 
        ("文字情緒辨識", test_text_emotion),
        ("臉部情緒辨識", test_facial_emotion),
        ("情緒融合", test_emotion_fusion),
        ("多模態整合", test_multi_modal)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n {test_name} 測試通過")
            else:
                print(f"\n {test_name} 測試失敗")
        except Exception as e:
            print(f"\n {test_name} 測試異常: {e}")
    
    print("\n" + "=" * 50)
    print(f"🏁 測試完成: {passed}/{total} 通過")
    
    if passed == total:
        print(" 所有測試通過！系統準備就緒。")
        return True
    else:
        print(" 部分測試失敗，請檢查相關模組。")
        return False

def main():
    """主函數"""
    try:
        success = run_all_tests()
        
        print("\n 下一步:")
        if success:
            print("1. 運行 python main.py 啟動完整系統")
            print("2. 運行 python emotion_api.py 啟動API服務")
            print("3. 開啟 emotion_chart.html 查看情緒圖表")
        else:
            print("1. 檢查缺少的套件並安裝")
            print("2. 確認配置文件設定正確")
            print("3. 重新執行測試")
            
    except KeyboardInterrupt:
        print("\n\n 測試中斷")
    except Exception as e:
        print(f"\n 測試執行異常: {e}")

if __name__ == "__main__":
    main()
