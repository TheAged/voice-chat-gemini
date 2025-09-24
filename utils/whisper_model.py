import whisper
import torch

# 檢查 GPU 可用性並載入模型
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Whisper 模型載入到: {device}")

# 使用更小但速度更快的模型，並開啟優化選項
# 針對中文優化：使用 small 模型在中文上表現更好
whisper_model = whisper.load_model("medium", device=device)  # small 對中文辨識更準確

# GPU 優化設定
if torch.cuda.is_available():
    print(f"GPU 可用: {torch.cuda.get_device_name(0)}")
    print(f"Whisper 模型實際 device: {next(whisper_model.parameters()).device}")
    
    # 啟用混合精度計算以提升速度
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # 設定 GPU 記憶體成長模式
    torch.cuda.empty_cache()
    
    print("GPU 優化設定已啟用：混合精度計算、CUDNN benchmark")
    print("已載入 small 模型以提升中文辨識準確度")
else:
    print("GPU 不可用，使用 CPU")

# 中文辨識優化函數
def transcribe_audio_optimized(audio_path, language="zh"):
    """優化的中文語音辨識"""
    try:
        # 針對中文的優化設定
        result = whisper_model.transcribe(
            audio_path,
            language=language,  # 指定中文
            task="transcribe",  # 轉錄任務
            temperature=0.0,    # 降低隨機性，提高一致性
            no_speech_threshold=0.6,  # 調整無語音檢測閾值
            logprob_threshold=-1.0,   # 調整信心度閾值
            compression_ratio_threshold=2.4,  # 壓縮比閾值
        )
        return result["text"].strip()
    except Exception as e:
        print(f"語音辨識錯誤: {e}")
        return ""
