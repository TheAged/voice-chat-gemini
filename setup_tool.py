#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情緒辨識系統模式切換工具
用於在開發模式和生產模式之間快速切換
"""

import sys
import os

# 加入當前目錄到路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def switch_mode():
    """模式切換主函數"""
    print("情緒辨識系統模式切換工具")
    print("=" * 40)
    
    try:
        from emotion_config import (
            USE_DEVELOPMENT_MODE, 
            switch_to_production_mode, 
            switch_to_development_mode,
            print_config_status
        )
        
        # 顯示當前狀態
        print(" 當前配置:")
        print_config_status()
        
        print("\n 選擇模式:")
        print("1. 開發模式 (Development Mode)")
        print("   - 臉部辨識使用模擬數據")
        print("   - 詳細除錯輸出")
        print("   - 適合在沒有攝影機的環境測試")
        print("")
        print("2. 生產模式 (Production Mode)")
        print("   - 啟用真實臉部辨識")
        print("   - 需要攝影機硬體")
        print("   - 適合部署到機器人")
        print("")
        print("3. 不改變，退出")
        
        while True:
            choice = input("\n請選擇 (1/2/3): ").strip()
            
            if choice == "1":
                print("\n 切換到開發模式...")
                modify_config_file(development=True)
                print(" 已切換到開發模式")
                print(" 重新啟動程式以生效")
                break
                
            elif choice == "2":
                print("\n 切換到生產模式...")
                print(" 注意: 生產模式需要安裝臉部辨識套件:")
                print("   pip install opencv-python fer")
                
                confirm = input("確定要切換到生產模式嗎? (y/N): ").strip().lower()
                if confirm in ['y', 'yes']:
                    modify_config_file(development=False)
                    print(" 已切換到生產模式")
                    print(" 重新啟動程式以生效")
                else:
                    print(" 取消切換")
                break
                
            elif choice == "3":
                print(" 退出，未做任何更改")
                break
                
            else:
                print(" 無效選擇，請輸入 1、2 或 3")
        
    except Exception as e:
        print(f" 模式切換失敗: {e}")

def modify_config_file(development=True):
    """修改配置檔案中的模式設定"""
    config_file = "emotion_config.py"
    
    try:
        # 讀取原始檔案
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替換模式設定
        new_value = str(development)
        if "USE_DEVELOPMENT_MODE = True" in content:
            content = content.replace("USE_DEVELOPMENT_MODE = True", f"USE_DEVELOPMENT_MODE = {new_value}")
        elif "USE_DEVELOPMENT_MODE = False" in content:
            content = content.replace("USE_DEVELOPMENT_MODE = False", f"USE_DEVELOPMENT_MODE = {new_value}")
        else:
            print(" 無法找到 USE_DEVELOPMENT_MODE 設定")
            return False
        
        # 寫回檔案
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
        
    except Exception as e:
        print(f" 配置檔案修改失敗: {e}")
        return False

def check_dependencies():
    """檢查相依套件安裝狀況"""
    print("\n 檢查相依套件...")
    
    required_packages = [
        "google.generativeai",
        "transformers", 
        "torch",
        "librosa",
        "whisper",
        "sounddevice",
        "scipy",
        "flask",
        "schedule",
        "emoji",
        "edge_tts"
    ]
    
    optional_packages = [
        "cv2",
        "fer"
    ]
    
    missing_required = []
    missing_optional = []
    
    # 檢查必要套件
    for package in required_packages:
        try:
            __import__(package)
            print(f" {package}")
        except ImportError:
            missing_required.append(package)
            print(f" {package} (必要)")
    
    # 檢查可選套件
    for package in optional_packages:
        try:
            __import__(package)
            print(f" {package} (臉部辨識)")
        except ImportError:
            missing_optional.append(package)
            print(f" {package} (臉部辨識，可選)")
    
    # 總結
    if missing_required:
        print(f"\n 缺少必要套件: {', '.join(missing_required)}")
        print("請執行: pip install -r requirements.txt")
        return False
    elif missing_optional:
        print(f"\n 缺少可選套件: {', '.join(missing_optional)}")
        print("開發模式可正常運行，生產模式需要安裝臉部辨識套件")
        print("安裝指令: pip install opencv-python fer")
        return True
    else:
        print("\n 所有套件已安裝完成")
        return True

def main():
    """主函數"""
    try:
        print(" 情緒辨識系統工具集")
        print("=" * 50)
        
        print("選擇功能:")
        print("1. 切換運行模式")
        print("2. 檢查套件安裝")
        print("3. 執行系統測試")
        print("4. 退出")
        
        while True:
            choice = input("\n請選擇功能 (1/2/3/4): ").strip()
            
            if choice == "1":
                switch_mode()
                break
            elif choice == "2":
                check_dependencies()
                break
            elif choice == "3":
                print("\n 執行系統測試...")
                os.system("python test_emotion_system.py")
                break
            elif choice == "4":
                print(" 再見！")
                break
            else:
                print(" 無效選擇，請輸入 1、2、3 或 4")
                
    except KeyboardInterrupt:
        print("\n\n 操作中斷")
    except Exception as e:
        print(f"\n 執行異常: {e}")

if __name__ == "__main__":
    main()
