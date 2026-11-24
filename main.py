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
import json, logging, math, os, re, shutil, subprocess, threading, uuid
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fastclip")

default_data = Path(os.path.expanduser("~/.fastclip/data"))
ROOT = Path(os.getenv("FASTCLIP_DATA", str(default_data)))
FRAMES_DIR = ROOT / "frames"
ROOT.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

COOKIES_FILE = Path(__file__).resolve().parent / "youtube_cookies.txt"
DENO_PATH = shutil.which("deno") or ("/opt/homebrew/bin/deno" if Path("/opt/homebrew/bin/deno").exists() else "deno")

def ensure_cookies():
    """Restores YouTube cookies from YOUTUBE_COOKIES_BASE64 environment variable if available."""
    b64 = os.getenv("YOUTUBE_COOKIES_BASE64")
    if b64:
        try:
            import base64
            decoded = base64.b64decode(b64.strip())
            if not COOKIES_FILE.exists() or COOKIES_FILE.stat().st_size < len(decoded):
                COOKIES_FILE.write_bytes(decoded)
                logger.info(f"Loaded {len(decoded)} bytes of YouTube cookies from environment.")
        except Exception as e:
            logger.warning(f"Failed to decode YOUTUBE_COOKIES_BASE64: {e}")

ensure_cookies()

def build_ytdlp_base_cmd() -> list[str]:
    ensure_cookies()
    cmd = [
        "yt-dlp",
        "--no-check-certificates",
        "--no-playlist",
        "--no-part",
        "--retries", "3",
        "--fragment-retries", "3",
        "--remote-components", "ejs:github",
        "--js-runtimes", f"deno:{DENO_PATH}",
        "--concurrent-fragments", "4"
    ]
    if COOKIES_FILE.exists() and COOKIES_FILE.stat().st_size > 100:
        cmd.extend(["--cookies", str(COOKIES_FILE)])
    else:
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

            # Streamer cam box (tight around facecam, centered on face with slight chest offset)
            box_w = 0.21
            box_h = 0.25
            box_x = max(0.0, min(1.0 - box_w, cx - box_w / 2.0))
            box_y = max(0.0, min(1.0 - box_h, cy - box_h / 2.0 + 0.03))

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
    """Selects Apple VideoToolbox GPU encoder locally (10x faster, sharp high-bitrate) or libx264 on Linux."""
    try:
        p = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=5)
        if "h264_videotoolbox" in p.stdout:
            if resolution == "4k":
                br, maxr, bufr = "45M", "65M", "90M"
            elif resolution == "2k":
                br, maxr, bufr = "30M", "45M", "60M"
            else:
                br, maxr, bufr = "22M", "32M", "44M"
            return (
                [
                    "-c:v", "h264_videotoolbox",
                    "-b:v", br,
                    "-maxrate", maxr,
                    "-bufsize", bufr,
                    "-profile:v", "high",
                    "-pix_fmt", "yuv420p",
                    "-r", "60",
                    "-fps_mode", "cfr",
                ],
                f"h264_videotoolbox-{br}-60fps",
            )
    except Exception:
        pass
    crf = "15" if resolution == "4k" else "16"
    return (
        [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", crf,
            "-profile:v", "high",
            "-level", "5.2",
            "-pix_fmt", "yuv420p",
            "-r", "60",
            "-fps_mode", "cfr",
        ],
        f"libx264-crf{crf}-60fps",
    )


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
    """Builds the FFmpeg video filter for the selected 9:16 layout at 60fps with Lanczos sharp scaling."""
    def to_even(v: int) -> int:
        return max(2, (int(v) // 2) * 2)

    def get_coord(box: Any, attr: str, default: float) -> float:
        if isinstance(box, dict):
            return float(box.get(attr, default))
        return float(getattr(box, attr, default))

    cx = to_even(get_coord(cam, "x", 0.05) * source_w)
    cy = to_even(get_coord(cam, "y", 0.18) * source_h)
    cw = to_even(min(get_coord(cam, "w", 0.26) * source_w, source_w - cx))
    ch = to_even(min(get_coord(cam, "h", 0.34) * source_h, source_h - cy))

    gx = to_even(get_coord(game, "x", 0.32) * source_w)
    gy = to_even(get_coord(game, "y", 0.05) * source_h)
    gw = to_even(min(get_coord(game, "w", 0.45) * source_w, source_w - gx))
    gh = to_even(min(get_coord(game, "h", 0.90) * source_h, source_h - gy))

    if layout == "split":
        if game_size == "50/50":
            ratio = 0.50
        elif game_size == "70/30":
            ratio = 0.70
        else:
            ratio = 0.60
        gh_out = to_even(target_h * ratio)
        ch_out = target_h - gh_out

        filter_str = (
            f"[0:v]crop={gw}:{gh}:{gx}:{gy},scale={target_w}:{gh_out}:force_original_aspect_ratio=increase:flags=lanczos,crop={target_w}:{gh_out}[top];"
            f"[0:v]crop={cw}:{ch}:{cx}:{cy},scale={target_w}:{ch_out}:force_original_aspect_ratio=increase:flags=lanczos,crop={target_w}:{ch_out}[bot];"
            f"[top][bot]vstack=inputs=2[v_pre]"
        )

    elif layout == "big_cam":
        ch_out = to_even(target_h * 0.60)
        gh_out = target_h - ch_out
        filter_str = (
            f"[0:v]crop={cw}:{ch}:{cx}:{cy},scale={target_w}:{ch_out}:force_original_aspect_ratio=increase:flags=lanczos,crop={target_w}:{ch_out}[top];"
            f"[0:v]crop={gw}:{gh}:{gx}:{gy},scale={target_w}:{gh_out}:force_original_aspect_ratio=increase:flags=lanczos,crop={target_w}:{gh_out}[bot];"
            f"[top][bot]vstack=inputs=2[v_pre]"
        )

    elif layout == "game_only":
        filter_str = f"[0:v]crop={gw}:{gh}:{gx}:{gy},scale={target_w}:{target_h}:force_original_aspect_ratio=increase:flags=lanczos,crop={target_w}:{target_h}[v_pre]"

    elif layout == "whole_screen":
        filter_str = (
            f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=increase:flags=lanczos,crop={target_w}:{target_h},boxblur=20:20[bg];"
            f"[0:v]scale={target_w}:-2:flags=lanczos[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[v_pre]"
        )

    elif layout == "cam_box":
        pip_w = to_even(target_w * 0.36)
        pip_h = to_even(pip_w * 0.75)
        filter_str = (
            f"[0:v]crop={gw}:{gh}:{gx}:{gy},scale={target_w}:{target_h}:force_original_aspect_ratio=increase:flags=lanczos,crop={target_w}:{target_h}[bg];"
            f"[0:v]crop={cw}:{ch}:{cx}:{cy},scale={pip_w}:{pip_h}:force_original_aspect_ratio=increase:flags=lanczos,crop={pip_w}:{pip_h}[cam];"
            f"[bg][cam]overlay=W-w-40:H-h-120[v_pre]"
        )

    elif layout == "circle_cam":
        pip_s = to_even(target_w * 0.35)
        filter_str = (
            f"[0:v]crop={gw}:{gh}:{gx}:{gy},scale={target_w}:{target_h}:force_original_aspect_ratio=increase:flags=lanczos,crop={target_w}:{target_h}[bg];"
            f"[0:v]crop={cw}:{ch}:{cx}:{cy},scale={pip_s}:{pip_s}:force_original_aspect_ratio=increase:flags=lanczos,crop={pip_s}:{pip_s}[cam];"
            f"[bg][cam]overlay=W-w-40:H-h-120[v_pre]"
        )

    elif layout == "cam_game":
        ch_out = to_even(target_h * 0.40)
        filter_str = (
            f"color=c=black:s={target_w}x{target_h}:r=60:d=9999[bg];"
            f"[0:v]crop={cw}:{ch}:{cx}:{cy},scale={target_w}:{ch_out}:force_original_aspect_ratio=increase:flags=lanczos,crop={target_w}:{ch_out}[cam];"
            f"[0:v]crop={gw}:{gh}:{gx}:{gy},scale={target_w}:-2:flags=lanczos[game];"
            f"[bg][cam]overlay=0:0:shortest=1[tmp];"
            f"[tmp][game]overlay=0:{ch_out + 40}:shortest=1[v_pre]"
        )

    else:
        gh_out = to_even(target_h * 0.60)
        ch_out = target_h - gh_out
        filter_str = (
            f"[0:v]crop={gw}:{gh}:{gx}:{gy},scale={target_w}:{gh_out}:force_original_aspect_ratio=increase:flags=lanczos,crop={target_w}:{gh_out}[top];"
            f"[0:v]crop={cw}:{ch}:{cx}:{cy},scale={target_w}:{ch_out}:force_original_aspect_ratio=increase:flags=lanczos,crop={target_w}:{ch_out}[bot];"
            f"[top][bot]vstack=inputs=2[v_pre]"
        )

    filter_str = f"{filter_str};[v_pre]fps=60[v_base]"
    return filter_str, "[v_base]"


def _ease_out_back(u: float) -> float:
    u = max(0.0, min(1.0, u))
    c1 = 1.70158
    c3 = c1 + 1.0
    p = u - 1.0
    return 1.0 + c3 * (p * p * p) + c1 * (p * p)


def _ease_out_cubic(u: float) -> float:
    u = max(0.0, min(1.0, u))
    p = 1.0 - u
    return 1.0 - p * p * p


def _build_word_chunks(transcript: list[dict], clip_start: float, clip_end: float) -> list[dict]:
    """Converts transcript segments + word timestamps into punchy 2-3 word caption cues."""
    all_words = []
    for seg in transcript:
        s_a = max(clip_start, float(seg.get("start", 0.0)))
        s_b = min(clip_end, float(seg.get("end", 0.0)))
        if s_b - s_a < 0.12:
            continue
        seg_words = seg.get("words") or []
        if seg_words:
            for w in seg_words:
                wa = max(clip_start, float(w.get("start", s_a))) - clip_start
                wb = min(clip_end, float(w.get("end", s_b))) - clip_start
                wt = to_hinglish(str(w.get("word", "")).strip())
                if wt and wb > wa:
                    all_words.append({"word": wt.upper(), "start": wa, "end": wb})
        else:
            raw = to_hinglish(str(seg.get("text", "")).strip())
            parts = raw.split()
            if not parts:
                continue
            dt = (s_b - s_a) / len(parts)
            for idx_w, pw in enumerate(parts):
                wa = (s_a + idx_w * dt) - clip_start
                wb = (s_a + (idx_w + 1) * dt) - clip_start
                all_words.append({"word": pw.upper(), "start": wa, "end": wb})

    cues = []
    cur: list[dict] = []
    cur_chars = 0
    for w in all_words:
        wlen = len(w["word"])
        if cur and (len(cur) >= 3 or cur_chars + wlen + 1 > 17 or (w["start"] - cur[-1]["end"]) > 0.65):
            cues.append({
                "start": cur[0]["start"],
                "end": max(cur[-1]["end"], cur[0]["start"] + 0.25),
                "words": cur,
            })
            cur = []
            cur_chars = 0
        cur.append(w)
        cur_chars += wlen + 1
    if cur:
        cues.append({
            "start": cur[0]["start"],
            "end": max(cur[-1]["end"], cur[0]["start"] + 0.25),
            "words": cur,
        })
    return cues


def generate_animated_captions_mov(
    transcript: list[dict],
    clip_start: float,
    clip_end: float,
    target_w: int,
    target_h: int,
    font_family: str,
    color_hex: str,
    position: str,
    animation: str,
    out_mov: Path
) -> tuple[bool, int, int, int]:
    """
    Renders a smooth 60fps transparent QuickTime RLE (.mov) caption strip using
    word-level timestamps, elastic spring/cubic easing, and single-overlay compositing.
    Returns (created, strip_render_w, strip_render_h, band_y_in_target).
    """
    cues = _build_word_chunks(transcript, clip_start, clip_end)
    if not cues:
        return False, 0, 0, 0

    dur = max(0.5, clip_end - clip_start)
    render_w = min(target_w, 1080)
    render_h = 260
    scale_to_target = target_h / 1920.0
    band_h_target = int(render_h * (target_w / float(render_w)))

    if position == "top":
        band_y = int(target_h * 0.11)
    elif position == "bottom":
        band_y = max(0, min(target_h - band_h_target, int(target_h * 0.82) - band_h_target // 2))
    else:
        band_y = max(0, int(target_h * 0.56) - band_h_target // 2)

    font_file = FONT_MAP.get(font_family.lower().replace(" ", ""), FONT_MAP["montserrat"])
    if not font_file.exists():
        font_file = FONT_MAP["montserrat"]

    base_size = 56
    pop_size = 64
    small_size = 44
    def _load_heavy_font(sz: int):
        f = ImageFont.truetype(str(font_file), sz)
        try:
            f.set_variation_by_axes([850])
        except Exception:
            pass
        return f

    try:
        font_base = _load_heavy_font(base_size)
        font_pop = _load_heavy_font(pop_size)
        font_small = _load_heavy_font(small_size)
    except Exception:
        font_base = font_pop = font_small = ImageFont.load_default()

    accent_rgb = parse_hex_color(color_hex, default=(243, 107, 22))
    accent_lum = (0.299 * accent_rgb[0] + 0.587 * accent_rgb[1] + 0.114 * accent_rgb[2]) / 255.0
    pill_text_rgb = (18, 18, 17) if accent_lum > 0.55 else (250, 249, 247)
    ink_light = (250, 249, 247)
    fps = 60
    total_frames = max(1, int(math.ceil(dur * fps)))
    empty_bytes = Image.new("RGBA", (render_w, render_h), (0, 0, 0, 0)).tobytes()

    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgba",
            "-s", f"{render_w}x{render_h}", "-r", str(fps), "-i", "-",
            "-c:v", "qtrle", str(out_mov)
        ],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    cue_ptr = 0
    cached_key = None
    cached_bytes = empty_bytes

    def pick_scaled_font(scale_val: float) -> ImageFont.FreeTypeFont:
        if scale_val >= 1.07:
            return font_pop
        if scale_val <= 0.88:
            return font_small
        return font_base

    try:
        for f_idx in range(total_frames):
            t = f_idx / float(fps)
            while cue_ptr < len(cues) and t > cues[cue_ptr]["end"]:
                cue_ptr += 1

            if cue_ptr >= len(cues) or t < cues[cue_ptr]["start"]:
                proc.stdin.write(empty_bytes)
                continue

            cue = cues[cue_ptr]
            c_a, c_b = cue["start"], cue["end"]
            words = cue["words"]

            # Active spoken word index
            active_idx = len(words) - 1
            for wi, w in enumerate(words):
                if t <= w["end"]:
                    active_idx = wi
                    break

            phrase_age = t - c_a
            phrase_left = c_b - t
            w_age = t - words[active_idx]["start"]

            # Smooth opacity envelope
            alpha = 1.0
            if phrase_age < 0.08:
                alpha = _ease_out_cubic(phrase_age / 0.08)
            elif phrase_left < 0.07:
                alpha = max(0.0, phrase_left / 0.07)

            # Determine if this frame is an invariant steady-state hold
            steady_phrase = phrase_age >= 0.18 and phrase_left >= 0.08
            steady_word = w_age >= 0.15
            state_key = (cue_ptr, active_idx, steady_phrase, steady_word)
            if steady_phrase and steady_word and state_key == cached_key:
                proc.stdin.write(cached_bytes)
                continue

            img = Image.new("RGBA", (render_w, render_h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)

            # Motion curve by style
            if animation == "bounce":
                enter_u = min(1.0, phrase_age / 0.16)
                scale_p = 0.78 + 0.22 * _ease_out_back(enter_u)
                y_shift = -int(22 * (1.0 - _ease_out_back(enter_u)))
            elif animation == "clean_cut":
                enter_u = min(1.0, phrase_age / 0.14)
                scale_p = 1.0
                y_shift = int(18 * (1.0 - _ease_out_cubic(enter_u)))
            else:
                # karaoke_pop
                enter_u = min(1.0, phrase_age / 0.10)
                scale_p = 0.92 + 0.08 * _ease_out_back(enter_u)
                y_shift = int(8 * (1.0 - _ease_out_cubic(enter_u)))

            cy = render_h // 2 + y_shift
            gap = 18

            # Measure each word width cleanly
            word_specs = []
            total_w = 0
            for wi, w in enumerate(words):
                is_act = (wi == active_idx)
                w_scale = scale_p
                if is_act and animation in ("karaoke_pop", "bounce"):
                    pop_u = min(1.0, max(0.0, w_age) / 0.14)
                    # Overshoot peak to ~1.16 then settle cleanly to 1.10
                    w_scale = scale_p * (1.0 + 0.12 * _ease_out_back(pop_u))
                wf = pick_scaled_font(w_scale)
                bbox = draw.textbbox((0, 0), w["word"], font=wf, stroke_width=5)
                ww = max(10, bbox[2] - bbox[0])
                wh = max(20, bbox[3] - bbox[1])
                word_specs.append((w["word"], wf, ww, wh, is_act))
                total_w += ww
            total_w += gap * max(0, len(word_specs) - 1)

            a_byte = int(max(0, min(255, round(alpha * 255))))

            # Clean dark pill backing for clean_cut / karaoke_pop readability
            if animation == "clean_cut":
                pad_x, pad_y = 30, 22
                pill_a = int(205 * alpha)
                border_a = int(45 * alpha)
                x0 = render_w // 2 - total_w // 2 - pad_x
                x1 = render_w // 2 + total_w // 2 + pad_x
                y0 = cy - 44 - pad_y // 2
                y1 = cy + 44 + pad_y // 2
                draw.rounded_rectangle([x0, y0, x1, y1], radius=22, fill=(18, 18, 17, pill_a), outline=(250, 249, 247, border_a), width=2)

            cursor_x = render_w // 2 - total_w // 2
            for w_text, wf, ww, wh, is_act in word_specs:
                wx = cursor_x + ww // 2
                if is_act and animation == "karaoke_pop":
                    # Vibrant highlighted pill behind the active word with high-contrast text
                    pill_x0 = cursor_x - 12
                    pill_x1 = cursor_x + ww + 12
                    pill_y0 = cy - wh // 2 - 12
                    pill_y1 = cy + wh // 2 + 12
                    draw.rounded_rectangle([pill_x0, pill_y0, pill_x1, pill_y1], radius=14, fill=(*accent_rgb, a_byte))
                    draw.text((wx, cy), w_text, font=wf, anchor="mm", fill=(*pill_text_rgb, a_byte))
                else:
                    # Two-pass rendering: outer shadow/outline first, crisp inner fill on top
                    draw.text(
                        (wx + 3, cy + 4),
                        w_text,
                        font=wf,
                        anchor="mm",
                        stroke_width=5,
                        stroke_fill=(0, 0, 0, int(195 * alpha)),
                        fill=(0, 0, 0, int(195 * alpha)),
                    )
                    draw.text(
                        (wx, cy),
                        w_text,
                        font=wf,
                        anchor="mm",
                        stroke_width=4,
                        stroke_fill=(12, 12, 12, a_byte),
                        fill=(12, 12, 12, a_byte),
                    )
                    txt_fill = (*accent_rgb, a_byte) if is_act else (*ink_light, a_byte)
                    draw.text((wx, cy), w_text, font=wf, anchor="mm", fill=txt_fill)
                cursor_x += ww + gap

            frame_bytes = img.tobytes()
            cached_key = state_key
            cached_bytes = frame_bytes
            proc.stdin.write(frame_bytes)

        proc.stdin.close()
        rc = proc.wait()
        if rc != 0 or not out_mov.exists():
            err_txt = proc.stderr.read().decode(errors="ignore")
            logger.warning(f"Caption mov encode error rc={rc}: {err_txt[-500:]}")
            return False, 0, 0, 0
        return True, render_w, render_h, band_y
    except Exception as ce:
        logger.warning(f"Caption strip generator failed: {ce}")
        try:
            proc.kill()
        except Exception:
            pass
        return False, 0, 0, 0


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
            # Download highest possible quality & 60fps from source (4K/1440p/1080p60) for razor-sharp crops
            ytdlp_cmd = build_ytdlp_base_cmd()
            ytdlp_cmd.extend([
                "-S", "res,fps,vcodec:h264",
                "-f", "bestvideo[fps>=50]+bestaudio/bestvideo+bestaudio/best",
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
                run(ytdlp_cmd, timeout=240)
            except Exception as dl_err:
                # Fallback if 60fps/merge fails
                try:
                    fallback_cmd = build_ytdlp_base_cmd() + [
                        "-f", "bestvideo+bestaudio/best",
                        "--merge-output-format", "mp4",
                        "-o", str(work / "source.%(ext)s"),
                    ]
                    if req.start_seconds is not None and req.end_seconds is not None:
                        fallback_cmd.extend(["--download-sections", f"*{req.start_seconds}-{req.end_seconds}"])
                    fallback_cmd.append(clean_url)
                    run(fallback_cmd, timeout=180)
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
                    word_list = []
                    if getattr(s, "words", None):
                        for w in s.words:
                            wt = to_hinglish(w.word.strip())
                            if wt:
                                word_list.append({"word": wt, "start": float(w.start), "end": float(w.end)})
                    transcript.append({
                        "start": float(s.start),
                        "end": float(s.end),
                        "text": clean_text,
                        "words": word_list,
                    })
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

                # Burn smooth 60fps word-by-word animated captions via a single strip overlay
                if req.captions_enabled and caption_rows:
                    caps_mov = work / f"caps_{idx:02d}.mov"
                    ok_caps, strip_w, _, band_y = generate_animated_captions_mov(
                        transcript=transcript,
                        clip_start=start,
                        clip_end=end,
                        target_w=target_w,
                        target_h=target_h,
                        font_family=req.caption_font,
                        color_hex=req.caption_color,
                        position=req.caption_position,
                        animation=req.caption_animation,
                        out_mov=caps_mov
                    )
                    if ok_caps:
                        cmd.extend(["-i", str(caps_mov)])
                        if strip_w != target_w:
                            filter_parts.append(f"[1:v]scale={target_w}:-2:flags=lanczos[cap_scaled]")
                            filter_parts.append(f"{cur_v}[cap_scaled]overlay=0:{band_y}:shortest=1[v_final]")
                        else:
                            filter_parts.append(f"{cur_v}[1:v]overlay=0:{band_y}:shortest=1[v_final]")
                        cur_v = "[v_final]"

                final_filter = ";".join(filter_parts)
                # Map the final video output at 60fps & high-bitrate audio
                cmd.extend([
                    "-filter_complex", final_filter,
                    "-map", cur_v,
                    "-map", "0:a:0?",
                    *enc_args,
                    "-c:a", "aac",
                    "-b:a", "320k",
                    "-ar", "48000",
                    "-movflags", "+faststart",
                    str(out)
                ])
                run(cmd, timeout=600)

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
            f"via {enc_name} at {req.resolution} 60fps 9:16 {req.layout}"
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
    ensure_cookies()
    return {
        "status": "ok",
        "service": "FastClip",
        "version": "0.3.0",
        "cookies_file_exists": COOKIES_FILE.exists(),
        "cookies_file_size": COOKIES_FILE.stat().st_size if COOKIES_FILE.exists() else 0,
        "env_b64_len": len(os.getenv("YOUTUBE_COOKIES_BASE64") or ""),
        "deno_path": DENO_PATH,
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


def safe_float(val: Any, default: float = 60.0) -> float:
    try:
        f = float(val)
        return default if math.isnan(f) or math.isinf(f) else f
    except Exception:
        return default


def _frame_is_hd(frame_path: Path) -> bool:
    """Returns True only if frame exists, is >30KB, and width >= 1000px."""
    if not frame_path.exists() or frame_path.stat().st_size < 30000:
        return False
    try:
        img = cv2.imread(str(frame_path))
        return img is not None and img.shape[1] >= 1000
    except Exception:
        return False


def _ensure_hd_frame(clean_url: str, stream_id: str, ts: float, frame_path: Path, web_thumb: str = "") -> None:
    """Extracts an HD (>=1000px) reference frame cleanly without temp-dir collisions."""
    if _frame_is_hd(frame_path):
        return

    cached_source = ROOT / "cache" / f"{stream_id}.mp4"
    if cached_source.exists():
        try:
            dur_str = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(cached_source)], timeout=10)
            if float(dur_str) >= ts:
                run(["ffmpeg", "-y", "-ss", str(ts), "-i", str(cached_source), "-vframes", "1", "-q:v", "2", str(frame_path)], timeout=20)
        except Exception as fe:
            logger.warning(f"Local ffmpeg extraction from cache failed: {fe}")

    if not _frame_is_hd(frame_path):
        temp_dir = ROOT / f"tmp_{stream_id}_{int(ts)}_{uuid.uuid4().hex[:6]}"
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)
            ytdlp_cmd = build_ytdlp_base_cmd() + [
                "-S", "res,vcodec:h264",
                "--download-sections", f"*{int(ts)}-{int(ts)+1}",
                "-f", "bestvideo[height>=720]/bestvideo/best",
                "-o", str(temp_dir / "sample.%(ext)s"),
                clean_url
            ]
            run(ytdlp_cmd, timeout=45)
            samples = [p for p in temp_dir.glob("sample.*") if p.is_file() and not p.name.endswith(".part")]
            if samples:
                run(["ffmpeg", "-y", "-i", str(samples[0]), "-vframes", "1", "-q:v", "2", str(frame_path)], timeout=20)
        except Exception as fe:
            logger.warning(f"Direct yt-dlp HD frame extract failed: {fe}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    if not frame_path.exists():
        for thumb_url in [
            web_thumb,
            f"https://img.youtube.com/vi/{stream_id}/maxresdefault.jpg",
            f"https://i.ytimg.com/vi/{stream_id}/hqdefault.jpg"
        ]:
            if thumb_url and download_url_file(thumb_url, frame_path):
                break


@app.post("/stream/frame")
def get_stream_frame(req: StreamFrameRequest):
    """Extracts or retrieves an HD reference frame at any stream timestamp."""
    clean_url = normalize_url(str(req.url or ""))
    stream_id = extract_video_id(clean_url)
    ts = max(1.0, safe_float(req.timestamp_seconds, 60.0))
    frame_name = f"{stream_id}_{int(ts)}.jpg"
    frame_path = FRAMES_DIR / frame_name

    _ensure_hd_frame(clean_url, stream_id, ts, frame_path)

    if not frame_path.exists():
        try:
            img = Image.new("RGB", (1920, 1080), (36, 36, 35))
            draw = ImageDraw.Draw(img)
            draw.text((960, 540), f"{stream_id} @ {int(ts)}s", fill=(243, 107, 22), anchor="mm")
            img.save(str(frame_path), format="JPEG")
        except Exception:
            pass

    cam_box, game_box = detect_streamer_face(frame_path) if frame_path.exists() else (None, None)
    detected = cam_box is not None
    if not cam_box:
        cam_box = {"x": 0.04, "y": 0.42, "w": 0.21, "h": 0.25}
    if not game_box:
        game_box = {"x": 0.30, "y": 0.05, "w": 0.45, "h": 0.90}

    return {
        "frame_url": f"/frames/{frame_name}" if frame_path.exists() else "",
        "timestamp_seconds": ts,
        "detected_face": detected,
        "cam_box": cam_box,
        "game_box": game_box
    }


@app.post("/stream/info")
def get_stream_info(req: StreamInfoRequest):
    """Fetches stream metadata and extracts an HD reference frame with AI face positioning."""
    clean_url = normalize_url(str(req.url or ""))
    stream_id = extract_video_id(clean_url)
    ts = max(10.0, safe_float(req.timestamp_seconds, 60.0))

    try:
        ytdlp_cmd = build_ytdlp_base_cmd() + ["--dump-json", clean_url]
        raw_json = run(ytdlp_cmd, timeout=45)
        meta = json.loads(raw_json)
        title = meta.get("title", "Livestream")
        duration = float(meta.get("duration") or 0.0)
        web_thumb = meta.get("thumbnail") or ""
        src_w = int(meta.get("width") or 1920)
        src_h = int(meta.get("height") or 1080)
    except Exception as e:
        logger.warning(f"yt-dlp metadata fetch failed, falling back to oEmbed: {e}")
        oembed = fetch_youtube_oembed(clean_url)
        title = oembed.get("title") or "Livestream"
        duration = 0.0
        web_thumb = oembed.get("thumbnail_url") or f"https://img.youtube.com/vi/{stream_id}/hqdefault.jpg"
        src_w, src_h = 1920, 1080

    frame_name = f"{stream_id}_{int(ts)}.jpg"
    frame_path = FRAMES_DIR / frame_name
    _ensure_hd_frame(clean_url, stream_id, ts, frame_path, web_thumb=web_thumb)

    cam_box, game_box = detect_streamer_face(frame_path) if frame_path.exists() else (None, None)
    detected = cam_box is not None
    if not cam_box:
        cam_box = {"x": 0.04, "y": 0.42, "w": 0.21, "h": 0.25}
    if not game_box:
        game_box = {"x": 0.30, "y": 0.05, "w": 0.45, "h": 0.90}

    return {
        "id": stream_id,
        "title": title,
        "duration": duration,
        "width": src_w,
        "height": src_h,
        "frame_url": f"/frames/{frame_name}" if frame_path.exists() else web_thumb,
        "detected_face": detected,
        "default_cam_box": cam_box,
        "default_game_box": game_box
    }


@app.get("/test/ytdlp")
def test_ytdlp(client: str = "android"):
    ensure_cookies()
    cmd = [
        "yt-dlp",
        "--no-check-certificates",
        "--no-playlist",
        "--extractor-args", f"youtube:player_client={client}",
        "--dump-json",
        "https://www.youtube.com/live/91c96JtcIFg"
    ]
    if COOKIES_FILE.exists() and COOKIES_FILE.stat().st_size > 100:
        cmd.extend(["--cookies", str(COOKIES_FILE)])
    p = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "returncode": p.returncode,
        "stdout": p.stdout[:500],
        "stderr": p.stderr[:1000],
        "cookies_exists": COOKIES_FILE.exists(),
        "cookies_size": COOKIES_FILE.stat().st_size if COOKIES_FILE.exists() else 0
    }


@app.api_route("/frames/{filename}", methods=["GET", "HEAD"])
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

