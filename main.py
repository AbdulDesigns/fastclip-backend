"""FastClip self-hosted processing API.
Transforms streams and videos into 9:16 vertical reels with automatic
and customizable streamer facecam & gameplay layout split.
Features:
- Fast section downloads without live-stream manifest stall
- Hardware-accelerated rendering (VideoToolbox / libx264 fast CRF 18 / 4K)
- OpenCV YuNet AI face detection for auto camera & gameplay box positioning
- Burned-in word-by-word animated captions (Karaoke Pop, Bounce, Clean Cut)
- Google Fonts support (Montserrat, Anton, Bangers, Bebas Neue)
- Indic transliteration for Hindi-to-Hinglish stream captions
"""
from __future__ import annotations
import json, math, os, re, shutil, subprocess, threading, uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

default_data = Path(os.path.expanduser("~/.fastclip/data"))
ROOT = Path(os.getenv("FASTCLIP_DATA", str(default_data)))
FRAMES_DIR = ROOT / "frames"
ROOT.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

COOKIES_FILE = Path(__file__).resolve().parent / "youtube_cookies.txt"
DENO_PATH = shutil.which("deno") or ("/opt/homebrew/bin/deno" if Path("/opt/homebrew/bin/deno").exists() else "deno")

def build_ytdlp_base_cmd() -> list[str]:
    cmd = [
        "yt-dlp",
        "--no-check-certificates",
        "--no-playlist",
        "--remote-components", "ejs:github",
        "--js-runtimes", f"deno:{DENO_PATH}",
        "--downloader-args", "ffmpeg_i:-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 2 -timeout 10000000",
        "--concurrent-fragments", "4"
    ]
    if COOKIES_FILE.exists() and COOKIES_FILE.stat().st_size > 500:
        cmd.extend(["--cookies", str(COOKIES_FILE)])
    else:
        # Fallback to local browser cookies if brave exists
        try:
            if Path("/Applications/Brave Browser.app").exists():
                cmd.extend(["--cookies-from-browser", "brave"])
        except Exception:
            pass
    return cmd


FONTS_DIR = Path(__file__).parent / "fonts"
MODELS_DIR = Path(__file__).parent / "models"
YUNET_MODEL = MODELS_DIR / "face_detection_yunet.onnx"

FONT_MAP = {
    "montserrat": FONTS_DIR / "Montserrat-Variable.ttf",
    "anton": FONTS_DIR / "Anton-Regular.ttf",
    "bangers": FONTS_DIR / "Bangers-Regular.ttf",
    "bebasneue": FONTS_DIR / "BebasNeue-Regular.ttf",
    "bebas neue": FONTS_DIR / "BebasNeue-Regular.ttf",
}

JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
app = FastAPI(title="FastClip Processing API", version="0.3.0")


class BoxRect(BaseModel):
    model_config = ConfigDict(extra="allow")
    x: Any = 0.0
    y: Any = 0.0
    w: Any = 1.0
    h: Any = 1.0


class ClipRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    url: Any = ""
    start_seconds: Any = None
    end_seconds: Any = None
    language: Any = "en"
    max_highlights: Any = 5
    layout: Any = "split" # "split", "circle_cam", "game_only", "whole_screen", "big_cam", "cam_box", "cam_game", "original"
    game_size: Any = "60/40" # "50/50", "60/40", "70/30"
    cam_box: Any = None
    game_box: Any = None
    captions_enabled: Any = True
    resolution: Any = "1080p" # "1080p", "2k", "4k"
    caption_font: Any = "Montserrat" # "Montserrat", "Anton", "Bangers", "BebasNeue"
    caption_color: Any = "#B7F36B" # Hex color
    caption_position: Any = "center" # "center", "top", "bottom"
    caption_animation: Any = "karaoke_pop" # "karaoke_pop", "bounce", "clean_cut"


class StreamFrameRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    url: Any = ""
    timestamp_seconds: Any = 60.0

class StreamInfoRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    url: Any = ""
    timestamp_seconds: Any = 60.0


def run(args: list[str], timeout: float = 180.0) -> str:
    env = dict(os.environ)
    env["PATH"] = f"/opt/homebrew/bin:/usr/local/bin:{env.get('PATH', '')}"
    try:
        p = subprocess.run(args, capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Command timed out after {timeout}s: {' '.join(args[:4])}")
    if p.returncode:
        raise RuntimeError((p.stderr or p.stdout)[-4000:])
    return p.stdout.strip()


def normalize_url(raw_url: str) -> str:
    url = str(raw_url or "").strip()
    if not url:
        return ""
    # Extract YouTube video ID if present (6 to 15 alphanumeric characters)
    yt_match = re.search(r"(?:v=|/live/|youtu\.be/|/shorts/|/embed/|^)([A-Za-z0-9_-]{6,15})(?:$|[?&/])", url)
    if yt_match:
        return f"https://www.youtube.com/watch?v={yt_match.group(1)}"
    if not url.startswith("http://") and not url.startswith("https://"):
        return f"https://{url}"
    return url


def extract_video_id(raw_url: str) -> str:
    m = re.search(r"(?:v=|/live/|youtu\.be/)([A-Za-z0-9_-]{11})", raw_url)
    if m:
        return m.group(1)
    return re.sub(r"[^a-zA-Z0-9_-]", "_", raw_url[-24:])


def parse_hex_color(hex_str: str, default=(183, 243, 107)) -> tuple[int, int, int]:
    clean = hex_str.lstrip("#")
    if len(clean) == 6:
        try:
            return tuple(int(clean[i:i+2], 16) for i in (0, 2, 4))
        except ValueError:
            pass
    return default


def to_hinglish(text: str) -> str:
    """Detects Devanagari Hindi characters and converts them to readable phonetic Latin script."""
    if re.search(r'[\u0900-\u097F]', text):
        try:
            from indic_transliteration import sanscript
            raw = sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
            clean = raw.lower().replace('aa', 'a').replace('ii', 'i').replace('uu', 'u')
            return clean.capitalize()
        except Exception as e:
            print(f"Hinglish transliteration warning: {e}")
    return text


def detect_streamer_face(frame_path: Path) -> tuple[dict | None, dict | None]:
    """Uses OpenCV YuNet ONNX model to detect streamer face and return optimal cam and game boxes."""
    if not frame_path.exists() or not YUNET_MODEL.exists():
        return None, None
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        frame = cv2.imread(str(frame_path))
        if frame is None:
            return None, None
        h, w = frame.shape[:2]
        detector = cv2.FaceDetectorYN.create(
            str(YUNET_MODEL),
            '',
            (w, h),
            0.45,
            0.3,
            5000,
            cv2.dnn.DNN_BACKEND_OPENCV,
            cv2.dnn.DNN_TARGET_CPU
        )
        detector.setInputSize((w, h))
        ret, faces = detector.detect(frame)
        if faces is not None and len(faces) > 0:
            # Pick face with highest confidence score
            best = max(faces, key=lambda f: f[-1])
            # Convert numpy types to native Python floats
            fx, fy, fw, fh = map(float, best[0:4])
            # Center of the face
            cx = (fx + fw / 2.0) / float(w)
            cy = (fy + fh / 2.0) / float(h)

            # Streamer cam box (width=0.28, height=0.36, centered on face with slight downward chest offset)
            box_w = 0.28
            box_h = 0.36
            box_x = max(0.0, min(1.0 - box_w, cx - box_w / 2.0))
            box_y = max(0.0, min(1.0 - box_h, cy - box_h / 2.0 + 0.04))

            # Gameplay box: place opposite the facecam
            if cx < 0.5:
                # Streamer is on left side -> gameplay is center-right
                game_x = 0.28
                game_y = 0.05
                game_w = 0.68
                game_h = 0.90
            else:
                # Streamer is on right side -> gameplay is left-center
                game_x = 0.04
                game_y = 0.05
                game_w = 0.68
                game_h = 0.90

            return (
                {"x": round(float(box_x), 3), "y": round(float(box_y), 3), "w": round(float(box_w), 3), "h": round(float(box_h), 3)},
                {"x": round(float(game_x), 3), "y": round(float(game_y), 3), "w": round(float(game_w), 3), "h": round(float(game_h), 3)}
            )
    except Exception as e:
        print(f"Face detector warning: {e}")
    return None, None


def get_best_video_encoder(resolution: str) -> tuple[list[str], str]:
    """Detects available hardware encoder (VideoToolbox on macOS) or falls back to fast libx264."""
    try:
        p = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
        if "h264_videotoolbox" in p.stdout:
            bitrate = "28M" if resolution == "4k" else ("18M" if resolution == "2k" else "14M")
            return (["-c:v", "h264_videotoolbox", "-b:v", bitrate, "-pix_fmt", "yuv420p"], "h264_videotoolbox")
    except Exception:
        pass
    crf = "17" if resolution == "4k" else "18"
    return (["-c:v", "libx264", "-preset", "veryfast", "-crf", crf, "-pix_fmt", "yuv420p"], "libx264")


def build_9_16_filter(
    source_w: int,
    source_h: int,
    layout: str,
    game_size: str,
    cam: BoxRect,
    game: BoxRect,
    target_w: int,
    target_h: int
) -> tuple[str, str]:
    """Builds the FFmpeg video filter for the selected 9:16 layout without aspect distortion."""
    def to_even(v: int) -> int:
        return max(2, (int(v) // 2) * 2)

    cx = to_even(cam.x * source_w)
    cy = to_even(cam.y * source_h)
    cw = to_even(min(cam.w * source_w, source_w - cx))
    ch = to_even(min(cam.h * source_h, source_h - cy))

    gx = to_even(game.x * source_w)
    gy = to_even(game.y * source_h)
    gw = to_even(min(game.w * source_w, source_w - gx))
    gh = to_even(min(game.h * source_h, source_h - gy))

    if layout == "split":
        # Game on top, Cam on bottom
        if game_size == "50/50":
            ratio = 0.50
        elif game_size == "70/30":
            ratio = 0.70
        else: # "60/40"
            ratio = 0.60
        gh_out = to_even(target_h * ratio)
        ch_out = target_h - gh_out

        filter_str = (
            f"[0:v]crop={gw}:{gh}:{gx}:{gy},scale={target_w}:{gh_out}:force_original_aspect_ratio=increase,crop={target_w}:{gh_out}[top];"
            f"[0:v]crop={cw}:{ch}:{cx}:{cy},scale={target_w}:{ch_out}:force_original_aspect_ratio=increase,crop={target_w}:{ch_out}[bot];"
            f"[top][bot]vstack=inputs=2[v_base]"
        )

    elif layout == "big_cam":
        # Cam on top 60%, Game on bottom 40%
        ch_out = to_even(target_h * 0.60)
        gh_out = target_h - ch_out
        filter_str = (
            f"[0:v]crop={cw}:{ch}:{cx}:{cy},scale={target_w}:{ch_out}:force_original_aspect_ratio=increase,crop={target_w}:{ch_out}[top];"
            f"[0:v]crop={gw}:{gh}:{gx}:{gy},scale={target_w}:{gh_out}:force_original_aspect_ratio=increase,crop={target_w}:{gh_out}[bot];"
            f"[top][bot]vstack=inputs=2[v_base]"
        )

    elif layout == "game_only":
        filter_str = f"[0:v]crop={gw}:{gh}:{gx}:{gy},scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h}[v_base]"

    elif layout == "whole_screen":
        filter_str = (
            f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h},boxblur=20:20[bg];"
            f"[0:v]scale={target_w}:-1[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[v_base]"
        )

    elif layout == "cam_box":
        pip_w = to_even(target_w * 0.36)
        pip_h = to_even(pip_w * 0.75)
        filter_str = (
            f"[0:v]crop={gw}:{gh}:{gx}:{gy},scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h}[bg];"
            f"[0:v]crop={cw}:{ch}:{cx}:{cy},scale={pip_w}:{pip_h}:force_original_aspect_ratio=increase,crop={pip_w}:{pip_h}[cam];"
            f"[bg][cam]overlay=W-w-40:H-h-120[v_base]"
        )

    elif layout == "circle_cam":
        pip_s = to_even(target_w * 0.35)
        filter_str = (
            f"[0:v]crop={gw}:{gh}:{gx}:{gy},scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h}[bg];"
            f"[0:v]crop={cw}:{ch}:{cx}:{cy},scale={pip_s}:{pip_s}:force_original_aspect_ratio=increase,crop={pip_s}:{pip_s}[cam];"
            f"[bg][cam]overlay=W-w-40:H-h-120[v_base]"
        )

    elif layout == "cam_game":
        ch_out = to_even(target_h * 0.40)
        filter_str = (
            f"color=c=black:s={target_w}x{target_h}:d=9999[bg];"
            f"[0:v]crop={cw}:{ch}:{cx}:{cy},scale={target_w}:{ch_out}:force_original_aspect_ratio=increase,crop={target_w}:{ch_out}[cam];"
            f"[0:v]scale={target_w}:-1[game];"
            f"[bg][cam]overlay=0:0:shortest=1[tmp];"
            f"[tmp][game]overlay=0:{ch_out + 40}:shortest=1[v_base]"
        )

    else:
        # Default split 60/40
        gh_out = to_even(target_h * 0.60)
        ch_out = target_h - gh_out
        filter_str = (
            f"[0:v]crop={gw}:{gh}:{gx}:{gy},scale={target_w}:{gh_out}:force_original_aspect_ratio=increase,crop={target_w}:{gh_out}[top];"
            f"[0:v]crop={cw}:{ch}:{cx}:{cy},scale={target_w}:{ch_out}:force_original_aspect_ratio=increase,crop={target_w}:{ch_out}[bot];"
            f"[top][bot]vstack=inputs=2[v_base]"
        )

    return filter_str, "[v_base]"


def generate_caption_overlays(
    transcript: list[dict],
    clip_start: float,
    clip_end: float,
    target_w: int,
    target_h: int,
    font_family: str,
    color_hex: str,
    position: str,
    animation: str,
    output_dir: Path
) -> list[tuple[Path, float, float]]:
    """Generates transparent PNG overlays with styled, transliterated, animated captions."""
    font_file = FONT_MAP.get(font_family.lower().replace(" ", ""), FONT_MAP["montserrat"])
    if not font_file.exists():
        font_file = FONT_MAP["montserrat"]

    scale_factor = target_h / 1920.0
    font_size = int(60 * scale_factor)
    stroke_w = int(7 * scale_factor)

    try:
        pil_font = ImageFont.truetype(str(font_file), font_size)
    except Exception:
        pil_font = ImageFont.load_default()

    primary_color = parse_hex_color(color_hex, default=(183, 243, 107))
    white_color = (255, 255, 255)
    outline_color = (0, 0, 0, 255)

    # Position Y coordinate
    if position == "top":
        base_y = int(target_h * 0.16)
    elif position == "bottom":
        base_y = int(target_h * 0.82)
    else: # "center" - right at the split boundary / focal area
        base_y = int(target_h * 0.58)

    overlays = []
    idx = 0

    for seg in transcript:
        s_start = max(clip_start, seg["start"])
        s_end = min(clip_end, seg["end"])
        if s_end - s_start < 0.25:
            continue

        raw_text = seg.get("text", "").strip()
        if not raw_text:
            continue

        # Transliterate Hindi to Hinglish if necessary
        text = to_hinglish(raw_text)

        # Break long sentences into punchy 3-5 word chunks
        words = text.split()
        if not words:
            continue

        chunk_size = 4
        dur_per_word = (s_end - s_start) / max(1, len(words))

        for w_idx in range(0, len(words), chunk_size):
            chunk_words = words[w_idx:w_idx + chunk_size]
            c_start = s_start + w_idx * dur_per_word - clip_start
            c_end = s_start + min(len(words), w_idx + chunk_size) * dur_per_word - clip_start

            if c_end <= c_start:
                continue

            chunk_text = " ".join(chunk_words).upper()
            png_path = output_dir / f"caption_{idx:03d}.png"

            img = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)

            # Apply animation visual tweaks
            y_pos = base_y
            if animation == "bounce":
                # Subtle bounce pop
                y_pos -= int(10 * scale_factor)

            # Draw drop shadow
            draw.text(
                (target_w // 2 + int(4 * scale_factor), y_pos + int(4 * scale_factor)),
                chunk_text,
                font=pil_font,
                anchor="mm",
                stroke_width=stroke_w,
                stroke_fill=(0, 0, 0, 200),
                fill=(0, 0, 0, 200)
            )

            # Draw text with thick outline
            # If karaoke_pop, use vibrant accent color; else use primary color
            fill_col = (*primary_color, 255) if (animation == "karaoke_pop" or idx % 2 == 0) else (*white_color, 255)
            draw.text(
                (target_w // 2, y_pos),
                chunk_text,
                font=pil_font,
                anchor="mm",
                stroke_width=stroke_w,
                stroke_fill=outline_color,
                fill=fill_col
            )

            img.save(str(png_path), format="PNG")
            overlays.append((png_path, c_start, c_end))
            idx += 1
            if idx >= 40: # FFmpeg input ceiling
                break
        if idx >= 40:
            break

    return overlays


def extract_audio_pcm(file_path: Path) -> np.ndarray:
    """Decodes media file audio to 16kHz float32 mono array via FFmpeg."""
    cmd = ["ffmpeg", "-y", "-i", str(file_path), "-f", "s16le", "-ac", "1", "-ar", "16000", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError("Failed to decode audio with ffmpeg")
    return np.frombuffer(p.stdout, np.int16).flatten().astype(np.float32) / 32768.0


def worker(job_id: str, req: ClipRequest):
    work = ROOT / job_id
    work.mkdir(parents=True, exist_ok=True)

    def update(**kw):
        with LOCK:
            JOBS[job_id].update(kw)

    try:
        t_start = datetime.now()
        update(status="downloading", progress=10)
        clean_url = normalize_url(str(req.url))
        stream_id = extract_video_id(clean_url)
        cache_dir = ROOT / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_source = cache_dir / f"{stream_id}.mp4"

        # Optimization 1: If cached full stream exists and covers timestamp, extract in <1s
        extracted_from_cache = False
        if cached_source.exists() and req.start_seconds is not None and req.end_seconds is not None:
            try:
                cdur_str = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(cached_source)], timeout=10)
                if float(cdur_str) >= req.end_seconds:
                    dur_sec = req.end_seconds - req.start_seconds
                    run(["ffmpeg", "-y", "-ss", str(req.start_seconds), "-i", str(cached_source), "-t", str(dur_sec), "-c", "copy", str(work / "source.mp4")], timeout=30)
                    extracted_from_cache = True
            except Exception as ce:
                print(f"Cache slice extraction notice: {ce}")

        if not extracted_from_cache:
            # Build yt-dlp arguments preferring reliable H.264 stream to prevent throttling/hangs
            ytdlp_cmd = build_ytdlp_base_cmd()
            ytdlp_cmd.extend([
                "-f", "bestvideo[vcodec^=avc1][height<=1080]+bestaudio[acodec^=mp4a]/bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
                "--merge-output-format", "mp4",
                "-o", str(work / "source.%(ext)s")
            ])

            if req.start_seconds is not None and req.end_seconds is not None:
                # Direct section fetch with socket reconnect to prevent hanging
                ytdlp_cmd.extend(["--download-sections", f"*{req.start_seconds}-{req.end_seconds}"])
            else:
                ytdlp_cmd.append("--live-from-start")

            ytdlp_cmd.append(clean_url)
            try:
                run(ytdlp_cmd, timeout=120)
            except Exception as dl_err:
                # Fast fallback with 720p or standard progressive
                try:
                    fallback_cmd = build_ytdlp_base_cmd() + [
                        "-f", "best[height<=720]/b/best",
                        "-o", str(work / "source.%(ext)s"),
                    ]
                    if req.start_seconds is not None and req.end_seconds is not None:
                        fallback_cmd.extend(["--download-sections", f"*{req.start_seconds}-{req.end_seconds}"])
                    fallback_cmd.append(clean_url)
                    run(fallback_cmd, timeout=90)
                except Exception:
                    raise dl_err

        sources = [p for p in work.glob("source.*") if p.is_file()]
        if not sources:
            raise RuntimeError("Downloader produced no source file")
        source = sources[0]
        # Only cache if full stream was downloaded (not a sliced section)
        if req.start_seconds is None and not cached_source.exists():
            try:
                shutil.copy2(str(source), str(cached_source))
            except Exception:
                pass
        t_dl = (datetime.now() - t_start).total_seconds()

        dur_str = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(source)])
        dur = float(dur_str)

        dims_str = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", str(source)])
        src_w, src_h = map(int, dims_str.strip().split("x")[:2])

        update(status="transcribing", progress=35, source_duration=dur, source_res=f"{src_w}x{src_h}")

        transcript = []
        if req.captions_enabled:
            try:
                from faster_whisper import WhisperModel
                audio_pcm = extract_audio_pcm(source)
                whisper_model_name = os.getenv("WHISPER_MODEL", "tiny")
                model = WhisperModel(whisper_model_name, device=os.getenv("WHISPER_DEVICE", "cpu"), compute_type=os.getenv("WHISPER_COMPUTE", "int8"))
                segments, info = model.transcribe(audio_pcm, language=req.language if req.language != "auto" else None, vad_filter=True, word_timestamps=True)
                for s in segments:
                    clean_text = to_hinglish(s.text.strip())
                    transcript.append({"start": float(s.start), "end": float(s.end), "text": clean_text})
            except Exception as we:
                print(f"Whisper transcription fallback: {we}")
                transcript = []
        (work / "transcript.json").write_text(json.dumps(transcript, ensure_ascii=False, indent=2))

        update(status="selecting", progress=60, transcript_segments=len(transcript))

        if req.start_seconds is not None and req.end_seconds is not None:
            ranges = [(0.0, dur, "Selected clip")]
        else:
            # Highlight ranking
            ranked = []
            cues = ("wow", "oh", "no way", "amazing", "insane", "incredible", "finally", "let's go", "laugh", "unbelievable", "what", "why")
            for i, s in enumerate(transcript):
                text = s["text"].lower()
                score = min(len(text) / 90.0, 1.5)
                score += sum(1.0 for cue in cues if cue in text)
                score += 0.35 * text.count("!") + 0.2 * text.count("?")
                a, b = max(0, s["start"] - 15), min(dur, s["end"] + 15)
                if b - a >= 6:
                    ranked.append((score, a, b, s["text"]))
            ranked.sort(reverse=True)
            chosen = []
            for score, a, b, label in ranked:
                if all(b <= x[1] or a >= x[2] for x in chosen):
                    chosen.append((score, a, b, label))
                if len(chosen) >= max(1, min(req.max_highlights, 20)):
                    break
            if not chosen:
                chosen = [(0, 0, min(dur, 60), "Opening segment")]
            ranges = [(a, b, label) for _, a, b, label in sorted(chosen, key=lambda x: x[1])]

        update(status="rendering", progress=75, clips_total=len(ranges))
        outputs = []

        # Resolution targets
        if req.resolution == "4k":
            target_w, target_h = 2160, 3840
        elif req.resolution == "2k":
            target_w, target_h = 1440, 2560
        else:
            target_w, target_h = 1080, 1920

        cam_b = req.cam_box or BoxRect(x=0.04, y=0.18, w=0.26, h=0.34)
        game_b = req.game_box or BoxRect(x=0.32, y=0.05, w=0.45, h=0.90)

        enc_args, enc_name = get_best_video_encoder(req.resolution)

        for idx, (start, end, label) in enumerate(ranges, 1):
            out = work / f"fastclip_{idx:02d}.mp4"
            srt = work / f"fastclip_{idx:02d}.srt"

            def stamp(seconds: float) -> str:
                ms = max(0, int(seconds * 1000))
                hh, rem = divmod(ms, 3600000)
                mm, rem = divmod(rem, 60000)
                ss, milli = divmod(rem, 1000)
                return f"{hh:02d}:{mm:02d}:{ss:02d},{milli:03d}"

            caption_rows = []
            for seg in transcript:
                a, b = max(start, seg["start"]), min(end, seg["end"])
                if b > a:
                    caption_rows.append((a - start, b - start, seg["text"]))
            srt.write_text("\n\n".join(f"{n}\n{stamp(a)} --> {stamp(b)}\n{text}" for n, (a, b, text) in enumerate(caption_rows, 1)), encoding="utf-8")

            if req.layout == "original":
                # Fast direct stream copy
                cmd = ["ffmpeg", "-y", "-ss", str(start), "-i", str(source), "-t", str(end - start),
                       "-c", "copy", "-movflags", "+faststart", str(out)]
                run(cmd)
            else:
                # 9:16 Vertical Reel conversion with burned-in subtitles
                base_filter, cur_v = build_9_16_filter(
                    source_w=src_w,
                    source_h=src_h,
                    layout=req.layout,
                    game_size=req.game_size,
                    cam=cam_b,
                    game=game_b,
                    target_w=target_w,
                    target_h=target_h
                )

                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(start),
                    "-i", str(source),
                    "-t", str(end - start)
                ]

                filter_parts = [base_filter]

                # Burn captions directly if enabled
                if req.captions_enabled and caption_rows:
                    caps_dir = work / f"caps_{idx:02d}"
                    caps_dir.mkdir(parents=True, exist_ok=True)
                    overlays = generate_caption_overlays(
                        transcript=transcript,
                        clip_start=start,
                        clip_end=end,
                        target_w=target_w,
                        target_h=target_h,
                        font_family=req.caption_font,
                        color_hex=req.caption_color,
                        position=req.caption_position,
                        animation=req.caption_animation,
                        output_dir=caps_dir
                    )

                    input_idx = 1
                    for png_path, c_a, c_b in overlays:
                        cmd.extend(["-i", str(png_path)])
                        next_v = f"[v_cap_{input_idx}]"
                        filter_parts.append(f"{cur_v}[{input_idx}:v]overlay=0:0:enable='between(t,{c_a:.3f},{c_b:.3f})'{next_v}")
                        cur_v = next_v
                        input_idx += 1

                final_filter = ";".join(filter_parts)
                # Map the final video output
                cmd.extend([
                    "-filter_complex", final_filter,
                    "-map", cur_v,
                    "-map", "0:a:0?",
                    *enc_args,
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-movflags", "+faststart",
                    str(out)
                ])
                run(cmd)

            outputs.append({
                "name": out.name,
                "label": label,
                "start": start,
                "end": end,
                "download": f"/jobs/{job_id}/files/{out.name}",
                "captions": f"/jobs/{job_id}/files/{srt.name}"
            })

        t_total = (datetime.now() - t_start).total_seconds()
        render_summary = (
            f"Done in {t_total:.1f}s (dl: {t_dl:.1f}s, render: {t_total - t_dl:.1f}s) "
            f"via {enc_name} at {req.resolution} 9:16 {req.layout}"
        )

        update(
            status="completed",
            progress=100,
            clips=outputs,
            finished_at=datetime.now(timezone.utc).isoformat(),
            render_summary=render_summary
        )
    except Exception as e:
        update(status="failed", error=str(e))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        raw = await request.body()
        body_text = raw.decode("utf-8", errors="ignore")
    except Exception:
        body_text = ""
    print(f"[RECOVERED 422] Request to {request.method} {request.url.path} failed validation: {exc.errors()}, raw_body={body_text}", flush=True)

    if request.url.path.endswith("/stream/info"):
        url = ""
        ts = 60.0
        try:
            parsed = json.loads(body_text) if body_text else {}
            url = str(parsed.get("url") or "")
            ts = float(parsed.get("timestamp_seconds") or 60.0)
        except Exception:
            pass
        return get_stream_info(StreamInfoRequest(url=url, timestamp_seconds=ts))

    if request.url.path.endswith("/stream/frame"):
        url = ""
        ts = 60.0
        try:
            parsed = json.loads(body_text) if body_text else {}
            url = str(parsed.get("url") or "")
            ts = float(parsed.get("timestamp_seconds") or 60.0)
        except Exception:
            pass
        return get_stream_frame(StreamFrameRequest(url=url, timestamp_seconds=ts))

    return JSONResponse(status_code=200, content={"status": "error", "detail": str(exc.errors())})


@app.get("/")
@app.get("/health")
@app.get("/healthz")
def health():
    return {
        "status": "ok",
        "service": "FastClip",
        "version": "0.3.0",
        "reel_layouts": ["split", "circle_cam", "game_only", "whole_screen", "big_cam", "cam_box", "cam_game"],
        "resolutions": ["1080p", "2k", "4k"],
        "fonts": ["Montserrat", "Anton", "Bangers", "BebasNeue"]
    }


def download_url_file(url: str, dest_path: Path) -> bool:
    """Safely downloads any URL directly using unverified SSL context and custom User-Agent."""
    try:
        import ssl, urllib.request
        ctx = ssl._create_unverified_context()
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        )
        with urllib.request.urlopen(req, context=ctx, timeout=8) as resp:
            data = resp.read()
            if len(data) > 500:
                dest_path.write_bytes(data)
                return True
    except Exception as e:
        print(f"Direct download failed for {url}: {e}")
    return False


def fetch_youtube_oembed(video_url: str) -> dict:
    """Queries oEmbed for video title and thumbnail without bot verification."""
    try:
        import ssl, urllib.request
        ctx = ssl._create_unverified_context()
        req = urllib.request.Request(
            f"https://noembed.com/embed?url={video_url}",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, context=ctx, timeout=5) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return {}


@app.post("/stream/frame")
def get_stream_frame(req: StreamFrameRequest):
    """Extracts or retrieves a reference frame at any stream timestamp with multi-tier fail-safe fallbacks."""
    clean_url = normalize_url(str(req.url))
    stream_id = extract_video_id(clean_url)
    ts = max(1.0, req.timestamp_seconds)
    frame_name = f"{stream_id}_{int(ts)}.jpg"
    frame_path = FRAMES_DIR / frame_name

    # If frame already exists with valid size (>30KB), return it immediately
    if not frame_path.exists() or frame_path.stat().st_size < 30000:
        # Priority 1: Check if local video source is cached and duration covers this timestamp
        cached_source = ROOT / "cache" / f"{stream_id}.mp4"
        if cached_source.exists():
            try:
                dur_str = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(cached_source)])
                if float(dur_str) >= ts:
                    run(["ffmpeg", "-y", "-ss", str(ts), "-i", str(cached_source), "-vframes", "1", "-q:v", "2", str(frame_path)])
            except Exception as fe:
                print(f"Local ffmpeg extraction from cache failed: {fe}")

    if not frame_path.exists() or frame_path.stat().st_size < 30000:
        # Priority 2: Direct yt-dlp section fetch at exact timestamp with cookies & JS runtime
        try:
            temp_dir = ROOT / f"tmp_frame_{stream_id}_{int(ts)}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            ytdlp_cmd = build_ytdlp_base_cmd() + [
                "--download-sections", f"*{int(ts)}-{int(ts)+1}",
                "-f", "bestvideo[height<=1080]/best[height<=1080]/best",
                "-o", str(temp_dir / "sample.%(ext)s"),
                clean_url
            ]
            run(ytdlp_cmd)
            samples = [p for p in temp_dir.glob("sample.*") if p.is_file() and not p.name.endswith(".part")]
            if samples:
                run(["ffmpeg", "-y", "-i", str(samples[0]), "-vframes", "1", "-q:v", "2", str(frame_path)])
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as fe:
            print(f"Direct yt-dlp frame extract failed: {fe}")

    if not frame_path.exists():
        # Priority 4: Direct YouTube maxresdefault / hqdefault download via SSL unverified HTTP
        for thumb_url in [
            f"https://img.youtube.com/vi/{stream_id}/maxresdefault.jpg",
            f"https://i.ytimg.com/vi/{stream_id}/hqdefault.jpg"
        ]:
            if download_url_file(thumb_url, frame_path):
                break

    if not frame_path.exists():
        # Priority 5: Fallback synthetic frame so API never fails with 500
        try:
            img = Image.new("RGB", (1920, 1080), (12, 16, 24))
            draw = ImageDraw.Draw(img)
            draw.text((960, 540), f"{stream_id} @ {int(ts)}s", fill=(183, 243, 107), anchor="mm")
            img.save(str(frame_path), format="JPEG")
        except Exception:
            pass

    cam_box, game_box = detect_streamer_face(frame_path) if frame_path.exists() else (None, None)
    if not cam_box:
        cam_box = {"x": 0.05, "y": 0.18, "w": 0.26, "h": 0.34}
    if not game_box:
        game_box = {"x": 0.32, "y": 0.05, "w": 0.45, "h": 0.90}

    return {
        "frame_url": f"/frames/{frame_name}" if frame_path.exists() else "",
        "timestamp_seconds": ts,
        "detected_face": cam_box is not None,
        "cam_box": cam_box,
        "game_box": game_box
    }


@app.post("/stream/info")
def get_stream_info(req: StreamInfoRequest):
    """Fetches stream metadata and extracts a representative frame with AI face positioning."""
    clean_url = normalize_url(str(req.url))
    stream_id = extract_video_id(clean_url)
    ts = max(10.0, req.timestamp_seconds)

    # 1. Inspect stream metadata via yt-dlp with oEmbed fallback
    try:
        raw_json = run([
            "yt-dlp",
            "--no-check-certificates",
            "--dump-json",
            "--no-playlist",
            "--js-runtimes", "node:/usr/local/bin/node",
            clean_url
        ])
        meta = json.loads(raw_json)
        title = meta.get("title", "Livestream")
        duration = float(meta.get("duration") or 0.0)
        web_thumb = meta.get("thumbnail") or ""
        src_w = int(meta.get("width") or 1920)
        src_h = int(meta.get("height") or 1080)
    except Exception as e:
        # Fallback to oEmbed for title
        oembed = fetch_youtube_oembed(clean_url)
        title = oembed.get("title") or "Livestream"
        duration = 0.0
        web_thumb = oembed.get("thumbnail_url") or f"https://img.youtube.com/vi/{stream_id}/hqdefault.jpg"
        src_w, src_h = 1920, 1080

    frame_name = f"{stream_id}_{int(ts)}.jpg"
    frame_path = FRAMES_DIR / frame_name

    if not frame_path.exists():
        # Check local cache for this stream ID first
        cached_source = ROOT / "cache" / f"{stream_id}.mp4"
        if cached_source.exists():
            try:
                run(["ffmpeg", "-y", "-ss", str(ts), "-i", str(cached_source), "-vframes", "1", "-q:v", "2", str(frame_path)])
            except Exception:
                pass

    if not frame_path.exists():
        try:
            temp_dir = ROOT / f"tmp_{stream_id}_{int(ts)}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            ytdlp_cmd = build_ytdlp_base_cmd() + [
                "--download-sections", f"*{int(ts)}-{int(ts)+1}",
                "-f", "bestvideo[height<=1080]/best[height<=1080]/best",
                "-o", str(temp_dir / "sample.%(ext)s"),
                clean_url
            ]
            run(ytdlp_cmd)
            samples = [p for p in temp_dir.glob("sample.*") if p.is_file() and not p.name.endswith(".part")]
            if samples:
                run(["ffmpeg", "-y", "-i", str(samples[0]), "-vframes", "1", "-q:v", "2", str(frame_path)])
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as fe:
            print(f"Direct frame download failed: {fe}")

    if not frame_path.exists():
        for thumb_url in [
            web_thumb,
            f"https://img.youtube.com/vi/{stream_id}/maxresdefault.jpg",
            f"https://i.ytimg.com/vi/{stream_id}/hqdefault.jpg"
        ]:
            if thumb_url and download_url_file(thumb_url, frame_path):
                break

    # Run AI face detection to auto-position camera and game bounding boxes
    cam_box, game_box = detect_streamer_face(frame_path) if frame_path.exists() else (None, None)

    # Fallbacks if face is not detected
    if not cam_box:
        cam_box = {"x": 0.05, "y": 0.18, "w": 0.26, "h": 0.34}
    if not game_box:
        game_box = {"x": 0.32, "y": 0.05, "w": 0.45, "h": 0.90}

    return {
        "id": stream_id,
        "title": title,
        "duration": duration,
        "width": src_w,
        "height": src_h,
        "frame_url": f"/frames/{frame_name}" if frame_path.exists() else web_thumb,
        "detected_face": cam_box is not None,
        "default_cam_box": cam_box,
        "default_game_box": game_box
    }


@app.get("/frames/{filename}")
def get_frame(filename: str):
    path = (FRAMES_DIR / filename).resolve()
    if path.parent != FRAMES_DIR.resolve() or not path.is_file():
        raise HTTPException(404, "Frame not found")
    return FileResponse(path, media_type="image/jpeg", filename=filename)


@app.post("/jobs")
def create_job(req: ClipRequest):
    job_id = uuid.uuid4().hex
    with LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": 0,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    threading.Thread(target=worker, args=(job_id, req), daemon=True).start()
    return JOBS[job_id]


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with LOCK:
        if job_id not in JOBS:
            raise HTTPException(404, "Job not found")
        return dict(JOBS[job_id])


@app.api_route("/jobs/{job_id}/files/{filename}", methods=["GET", "HEAD"])
def get_file(job_id: str, filename: str):
    path = (ROOT / job_id / filename).resolve()
    if path.parent != (ROOT / job_id).resolve() or not path.is_file() or path.suffix.lower() not in (".mp4", ".srt"):
        raise HTTPException(404, "File not found")
    media_type = "video/mp4" if path.suffix.lower() == ".mp4" else "application/x-subrip"
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.api_route("/apk", methods=["GET", "HEAD"])
@app.api_route("/download/apk", methods=["GET", "HEAD"])
def download_latest_apk():
    apk_candidates = [
        Path("/Users/abdul/Downloads/YT Clipping/FastClip-v0.3.0.apk"),
        Path("/Users/abdul/Downloads/YT Clipping/FastClip/app/build/outputs/apk/debug/app-debug.apk")
    ]
    for apk in apk_candidates:
        if apk.is_file():
            return FileResponse(apk, media_type="application/vnd.android.package-archive", filename="FastClip-v0.3.0.apk")
    raise HTTPException(404, "APK not found")

