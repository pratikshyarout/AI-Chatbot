#media_processing.py
import os
import cv2
import pytesseract
from PIL import Image
from faster_whisper import WhisperModel
from moviepy.editor import VideoFileClip

# Optional: point Tesseract if needed on Windows
# pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

_WHISPER = None

def _ensure_whisper():
    global _WHISPER
    if _WHISPER is None:
        # choose small for speed; change to "base" or "small" if you prefer
        _WHISPER = WhisperModel("tiny", device="cpu", compute_type="int8")
    return _WHISPER


def extract_text_from_image(path: str) -> str:
    try:
        img = Image.open(path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        return f"[image OCR failed: {e}]"


def extract_text_from_audio(path: str) -> str:
    try:
        model = _ensure_whisper()
        segments, info = model.transcribe(path, beam_size=1)
        return " ".join(seg.text for seg in segments).strip()
    except Exception as e:
        return f"[audio transcription failed: {e}]"


def _sample_video_frames(path: str, every_n_seconds: int = 5, max_frames: int = 6):
    frames_text = []
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return frames_text
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = int(fps * every_n_seconds)
        idx = 0
        grabbed = 0
        while True and grabbed < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            # OCR the frame after converting to RGB PIL
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            text = pytesseract.image_to_string(pil)
            if text.strip():
                frames_text.append(text.strip())
            idx += step
            grabbed += 1
        cap.release()
    except Exception:
        pass
    return frames_text


def extract_text_from_video(path: str) -> str:
    # 1) OCR a few frames for on-screen text
    ocr_chunks = _sample_video_frames(path)
    # 2) Extract audio -> transcribe
    audio_text = ""
    try:
        clip = VideoFileClip(path)
        audio_path = path + ".wav"
        clip.audio.write_audiofile(audio_path, logger=None)
        audio_text = extract_text_from_audio(audio_path)
        try:
            os.remove(audio_path)
        except Exception:
            pass
    except Exception as e:
        audio_text = f"[video audio extract failed: {e}]"
    parts = []
    if ocr_chunks:
        parts.append("OCR:\n" + "\n".join(ocr_chunks))
    if audio_text:
        parts.append("Audio:\n" + audio_text)
    return "\n\n".join(parts).strip()


def extract_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        return extract_text_from_image(path)
    if ext in [".wav", ".mp3", ".m4a", ".flac", ".ogg"]:
        return extract_text_from_audio(path)
    if ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        return extract_text_from_video(path)
    return ""
