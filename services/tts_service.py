# 文字轉語音服務 由凱比端負責

# pip install edge-tts
import edge_tts
import asyncio
import tempfile, os
import subprocess
import platform

class TTSService:
    def __init__(self, voice: str = "zh-TW-HsiaoChenNeural", rate: str = "+0%", pitch: str = "+16Hz"):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch

    async def synthesize_async(self, text: str) -> bytes:
        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate, pitch=self.pitch)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp_path = tmp.name
        try:
            await communicate.save(tmp_path)
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    def synthesize(self, text: str) -> bytes:
        return asyncio.run(self.synthesize_async(text))
    
    async def synthesize_and_play(self, text: str) -> bytes:
        """生成音頻並直接播放"""
        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate, pitch=self.pitch)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp_path = tmp.name
        try:
            await communicate.save(tmp_path)
            
            # 讀取音頻數據
            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()
            
            # 播放音頻
            await self._play_audio_file(tmp_path)
            
            return audio_bytes
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    
    async def _play_audio_file(self, file_path: str):
        """播放音頻檔案"""
        try:
            system = platform.system()
            if system == "Linux":
                # Linux 使用 mpg123 或 aplay
                try:
                    subprocess.run(["mpg123", file_path], check=True, capture_output=True)
                except subprocess.CalledProcessError:
                    try:
                        subprocess.run(["aplay", file_path], check=True, capture_output=True)
                    except subprocess.CalledProcessError:
                        print(f"無法播放音頻檔案: {file_path}，請安裝 mpg123 或 aplay")
            elif system == "Darwin":  # macOS
                subprocess.run(["afplay", file_path], check=True, capture_output=True)
            elif system == "Windows":
                import winsound
                winsound.PlaySound(file_path, winsound.SND_FILENAME)
            else:
                print(f"不支援的作業系統: {system}")
        except Exception as e:
            print(f"播放音頻失敗: {e}")
           
