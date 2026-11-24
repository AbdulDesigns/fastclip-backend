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
import concurrent.futures, importlib.util, json, logging, math, os, platform, re, shutil, subprocess, threading, time, uuid
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
    caption_font_size: Any = 19 # 14..32 relative units (size * 3 px on 1080w)
    game_name: Any = ""
    streamer_name: Any = ""
    encoder_mode: Any = "auto" # "auto", "gpu", "cpu"


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


HINGLISH_INITIAL_PROMPT = (
    "Haan bhai kya scene hai, aaj ki stream mein op clutch hone wala hai. "
    "Sab log ready raho, let's go! Bhai bilkul sahi kaha, maza aa gaya."
)

_URDU_MAP = {
    "ا": "a", "آ": "aa", "ب": "b", "پ": "p", "ت": "t", "ٹ": "t", "ث": "s",
    "ج": "j", "چ": "ch", "ح": "h", "خ": "kh", "د": "d", "ڈ": "d", "ذ": "z",
    "ر": "r", "ڑ": "r", "ز": "z", "ژ": "zh", "س": "s", "ش": "sh", "ص": "s",
    "ض": "z", "ط": "t", "ظ": "z", "ع": "a", "غ": "gh", "ف": "f", "ق": "q",
    "ک": "k", "گ": "g", "ل": "l", "م": "m", "ن": "n", "ں": "n", "و": "o",
    "ہ": "h", "ھ": "h", "ء": "", "ی": "y", "ے": "e", "ئ": "i", "ؤ": "o",
    "َ": "a", "ِ": "i", "ُ": "u", "ً": "an", "۔": ".", "،": ",", "؟": "?",
}

_WHISPER_LOCK = threading.Lock()
_FW_MODELS: dict[str, Any] = {}
_MLX_SNAPSHOT_CACHE: dict[str, str] = {}
_ENCODER_STDOUT_CACHE: str | None = None
_FONT_CACHE: dict[tuple[str, int], Any] = {}


def resolve_mlx_model_path(repo_id: str) -> str:
    """Resolves locally cached HF snapshot dir to avoid 4-15s Hub HTTP roundtrips per job."""
    if not repo_id or Path(repo_id).exists():
        return repo_id
    if repo_id in _MLX_SNAPSHOT_CACHE:
        return _MLX_SNAPSHOT_CACHE[repo_id]
    try:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(repo_id=repo_id, local_files_only=True)
        if local_dir and Path(local_dir).is_dir():
            _MLX_SNAPSHOT_CACHE[repo_id] = str(local_dir)
            return str(local_dir)
    except Exception:
        pass
    return repo_id


def warmup_runtime_async():
    """Pre-warms encoder probe and local Whisper model in background so first clip renders fast."""
    def _warm():
        try:
            get_best_video_encoder("1080p", encoder_mode="auto")
            if select_whisper_backend() == "mlx":
                import mlx_whisper
                repo = resolve_mlx_model_path(os.getenv("WHISPER_MLX_REPO", "mlx-community/whisper-small-mlx"))
                with _WHISPER_LOCK:
                    mlx_whisper.transcribe(
                        np.zeros(16000, dtype=np.float32),
                        path_or_hf_repo=repo,
                        language="en",
                    )
        except Exception as we:
            logger.debug(f"Background runtime warmup skipped/deferred: {we}")

    threading.Thread(target=_warm, daemon=True).start()


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in ("arm64", "aarch64")


def select_whisper_backend(backend_override: str | None = None) -> str:
    pref = (backend_override or os.getenv("FASTCLIP_WHISPER_BACKEND", "auto")).strip().lower()
    if pref == "faster-whisper" or pref == "cpu":
        return "faster-whisper"
    if pref == "mlx" and importlib.util.find_spec("mlx_whisper") is not None:
        return "mlx"
    if is_apple_silicon() and importlib.util.find_spec("mlx_whisper") is not None:
        return "mlx"
    return "faster-whisper"


def normalize_caption_language(language: Any = "en", captions_enabled: Any = True) -> str:
    if isinstance(captions_enabled, str):
        enabled = captions_enabled.strip().lower() not in ("false", "0", "no", "off", "")
    else:
        enabled = bool(captions_enabled)
    lang = str(language or "en").strip().lower()
    if not enabled or lang in ("off", "none", "disabled"):
        return "off"
    if lang in ("hinglish", "hi", "hi-en", "hi_en"):
        return "hinglish"
    return "en"


def normalize_caption_animation(animation: Any = "karaoke_pop") -> str:
    raw = str(animation or "karaoke_pop").strip().lower()
    if raw in ("standard", "clean", "clean_cut", "plain"):
        return "clean_cut"
    if raw in ("bounce",):
        return "bounce"
    return "karaoke_pop"


def to_hinglish(text: str) -> str:
    """Converts any stray Devanagari or Urdu glyphs to casual phonetic Roman Hinglish."""
    if not text:
        return ""
    out = text
    if re.search(r"[\u0900-\u097F]", out):
        try:
            from indic_transliteration import sanscript
            raw = sanscript.transliterate(out, sanscript.DEVANAGARI, sanscript.ITRANS)
            out = (
                raw.replace(".n", "n")
                .replace("~n", "n")
                .replace("N^", "n")
                .replace(".N", "n")
                .replace("M", "n")
                .replace("A", "aa")
                .replace("I", "ee")
                .replace("U", "oo")
                .replace("R^i", "ri")
                .replace("Ch", "chh")
                .replace("Sh", "sh")
            )
            out = re.sub(r"[~^.]", "", out)
        except Exception as e:
            logger.warning(f"Hinglish Devanagari transliteration warning: {e}")
    if re.search(r"[\u0600-\u06FF]", out):
        out = "".join(_URDU_MAP.get(ch, ch) for ch in out)
    # Collapse excess whitespace caused by stripped combining marks
    return re.sub(r"\s{2,}", " ", out).strip()


def transcribe_audio_pcm(
    audio_pcm: np.ndarray,
    caption_language: str = "en",
    backend_override: str | None = None,
    model_override: str | None = None,
) -> tuple[list[dict], dict]:
    """
    Transcribes 16kHz float32 PCM audio with word-level timestamps.
    On Apple Silicon prefers mlx-whisper (Metal GPU), falling back to cached faster-whisper CPU.
    For 'hinglish', uses language='en' + HINGLISH_INITIAL_PROMPT (100% code-switch retention
    vs 'language=hi' which drops English phrases) plus Roman cleanup.
    """
    lang_mode = normalize_caption_language(caption_language, captions_enabled=(caption_language != "off"))
    if lang_mode == "off" or audio_pcm.size == 0:
        return [], {"engine": "off", "model": "none", "language": "off", "elapsed_s": 0.0}

    prompt = HINGLISH_INITIAL_PROMPT if lang_mode == "hinglish" else None
    whisper_lang = "en"
    backend = select_whisper_backend(backend_override)
    t0 = time.perf_counter()

    def _finalize(seg_iter, get_text, get_start, get_end, get_words) -> list[dict]:
        rows: list[dict] = []
        for s in seg_iter:
            raw_t = str(get_text(s) or "").strip()
            txt = to_hinglish(raw_t) if lang_mode == "hinglish" else raw_t
            w_list: list[dict] = []
            for w in (get_words(s) or []):
                wt_raw = str(w["word"] if isinstance(w, dict) else getattr(w, "word", "")).strip()
                wt = to_hinglish(wt_raw) if lang_mode == "hinglish" else wt_raw
                ws = float(w["start"] if isinstance(w, dict) else getattr(w, "start", 0.0))
                we = float(w["end"] if isinstance(w, dict) else getattr(w, "end", ws))
                if wt:
                    w_list.append({"word": wt, "start": ws, "end": max(ws + 0.02, we)})
            if txt or w_list:
                rows.append({
                    "start": float(get_start(s)),
                    "end": float(get_end(s)),
                    "text": txt,
                    "words": w_list,
                })
        return rows

    if backend == "mlx":
        mlx_repo = model_override or os.getenv("WHISPER_MLX_REPO", "mlx-community/whisper-small-mlx")
        mlx_path = resolve_mlx_model_path(mlx_repo)
        try:
            import mlx_whisper
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
            with _WHISPER_LOCK:
                res = mlx_whisper.transcribe(
                    audio_pcm,
                    path_or_hf_repo=mlx_path,
                    language=whisper_lang,
                    initial_prompt=prompt,
                    word_timestamps=True,
                )
            transcript = _finalize(
                res.get("segments", []),
                lambda s: s.get("text", ""),
                lambda s: s.get("start", 0.0),
                lambda s: s.get("end", 0.0),
                lambda s: s.get("words") or [],
            )
            dt = round(time.perf_counter() - t0, 3)
            return transcript, {"engine": "mlx-whisper", "model": mlx_repo, "language": lang_mode, "elapsed_s": dt}
        except Exception as mlx_err:
            logger.warning(f"mlx-whisper fallback to faster-whisper CPU: {mlx_err}")

    from faster_whisper import WhisperModel
    fw_name = model_override or os.getenv("WHISPER_MODEL", "small")
    device = os.getenv("WHISPER_DEVICE", "cpu")
    compute = os.getenv("WHISPER_COMPUTE", "int8")
    cache_key = f"{fw_name}:{device}:{compute}"
    with _WHISPER_LOCK:
        model = _FW_MODELS.get(cache_key)
        if model is None:
            model = WhisperModel(fw_name, device=device, compute_type=compute)
            _FW_MODELS[cache_key] = model
        dur_s = float(len(audio_pcm)) / 16000.0
        segments, _info = model.transcribe(
            audio_pcm,
            language=whisper_lang,
            initial_prompt=prompt,
            vad_filter=(dur_s >= 15.0),
            word_timestamps=True,
        )
        segs_list = list(segments)

    transcript = _finalize(
        segs_list,
        lambda s: s.text,
        lambda s: s.start,
        lambda s: s.end,
        lambda s: getattr(s, "words", None) or [],
    )
    dt = round(time.perf_counter() - t0, 3)
    return transcript, {"engine": "faster-whisper", "model": fw_name, "language": lang_mode, "elapsed_s": dt}


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


def normalize_encoder_mode(mode: Any = "auto") -> str:
    m = str(mode or "auto").strip().lower()
    if m in ("cpu", "software", "libx264", "x264"):
        return "cpu"
    if m in ("gpu", "videotoolbox", "hw", "hardware"):
        return "gpu"
    return "auto"


def get_best_video_encoder(
    resolution: str,
    target_fps: int = 60,
    encoder_mode: Any = "auto",
    probe_encoders_fn=None,
) -> tuple[list[str], str]:
    """
    Selects encoder according to encoder_mode ('auto' | 'gpu' | 'cpu').
    Prefers Apple h264_videotoolbox (2.2x-3.7x faster, SW decode avoids copyback loss),
    and falls back safely to libx264 when CPU is chosen or GPU is unavailable.
    """
    fps_val = max(60, int(target_fps))
    mode = normalize_encoder_mode(encoder_mode)
    color_tags = [
        "-pix_fmt", "yuv420p",
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-color_range", "tv",
        "-r", str(fps_val),
        "-fps_mode", "cfr",
    ]
    if mode != "cpu":
        try:
            global _ENCODER_STDOUT_CACHE
            if probe_encoders_fn is not None:
                enc_stdout = str(probe_encoders_fn() or "")
            else:
                if _ENCODER_STDOUT_CACHE is None:
                    p = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=5)
                    _ENCODER_STDOUT_CACHE = p.stdout
                enc_stdout = _ENCODER_STDOUT_CACHE
            if "h264_videotoolbox" in enc_stdout:
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
                        *color_tags,
                    ],
                    f"h264_videotoolbox-{br}-{fps_val}fps",
                )
        except Exception as e:
            logger.warning(f"VideoToolbox probe failed, falling back to CPU libx264: {e}")

    crf = "15" if resolution == "4k" else "16"
    return (
        [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", crf,
            "-profile:v", "high",
            "-level", "5.2",
            *color_tags,
        ],
        f"libx264-crf{crf}-{fps_val}fps",
    )


def build_9_16_filter(
    source_w: int,
    source_h: int,
    layout: str,
    game_size: str,
    cam: BoxRect,
    game: BoxRect,
    target_w: int,
    target_h: int,
    target_fps: int = 60,
) -> tuple[str, str]:
    """Builds the FFmpeg video filter for 9:16 layout with accurate Lanczos scaling, SAR 1:1, and >=60fps preservation."""
    fps_val = max(60, int(target_fps))
    scale_flags = "lanczos+accurate_rnd+full_chroma_int"

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
            f"[0:v]crop={gw}:{gh}:{gx}:{gy},scale={target_w}:{gh_out}:force_original_aspect_ratio=increase:flags={scale_flags},crop={target_w}:{gh_out}[top];"
            f"[0:v]crop={cw}:{ch}:{cx}:{cy},scale={target_w}:{ch_out}:force_original_aspect_ratio=increase:flags={scale_flags},crop={target_w}:{ch_out}[bot];"
            f"[top][bot]vstack=inputs=2[v_pre]"
        )

    elif layout == "big_cam":
        ch_out = to_even(target_h * 0.60)
        gh_out = target_h - ch_out
        filter_str = (
            f"[0:v]crop={cw}:{ch}:{cx}:{cy},scale={target_w}:{ch_out}:force_original_aspect_ratio=increase:flags={scale_flags},crop={target_w}:{ch_out}[top];"
            f"[0:v]crop={gw}:{gh}:{gx}:{gy},scale={target_w}:{gh_out}:force_original_aspect_ratio=increase:flags={scale_flags},crop={target_w}:{gh_out}[bot];"
            f"[top][bot]vstack=inputs=2[v_pre]"
        )

    elif layout == "game_only":
        filter_str = f"[0:v]crop={gw}:{gh}:{gx}:{gy},scale={target_w}:{target_h}:force_original_aspect_ratio=increase:flags={scale_flags},crop={target_w}:{target_h}[v_pre]"

    elif layout == "whole_screen":
        filter_str = (
            f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=increase:flags={scale_flags},crop={target_w}:{target_h},boxblur=20:20[bg];"
            f"[0:v]scale={target_w}:-2:flags={scale_flags}[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[v_pre]"
        )

    elif layout == "cam_box":
        pip_w = to_even(target_w * 0.36)
        pip_h = to_even(pip_w * 0.75)
        filter_str = (
            f"[0:v]crop={gw}:{gh}:{gx}:{gy},scale={target_w}:{target_h}:force_original_aspect_ratio=increase:flags={scale_flags},crop={target_w}:{target_h}[bg];"
            f"[0:v]crop={cw}:{ch}:{cx}:{cy},scale={pip_w}:{pip_h}:force_original_aspect_ratio=increase:flags={scale_flags},crop={pip_w}:{pip_h}[cam];"
            f"[bg][cam]overlay=W-w-40:H-h-120[v_pre]"
        )

    elif layout == "circle_cam":
        pip_s = to_even(target_w * 0.35)
        filter_str = (
            f"[0:v]crop={gw}:{gh}:{gx}:{gy},scale={target_w}:{target_h}:force_original_aspect_ratio=increase:flags={scale_flags},crop={target_w}:{target_h}[bg];"
            f"[0:v]crop={cw}:{ch}:{cx}:{cy},scale={pip_s}:{pip_s}:force_original_aspect_ratio=increase:flags={scale_flags},crop={pip_s}:{pip_s}[cam];"
            f"[bg][cam]overlay=W-w-40:H-h-120[v_pre]"
        )

    elif layout == "cam_game":
        ch_out = to_even(target_h * 0.40)
        filter_str = (
            f"color=c=black:s={target_w}x{target_h}:r={fps_val}:d=9999[bg];"
            f"[0:v]crop={cw}:{ch}:{cx}:{cy},scale={target_w}:{ch_out}:force_original_aspect_ratio=increase:flags={scale_flags},crop={target_w}:{ch_out}[cam];"
            f"[0:v]crop={gw}:{gh}:{gx}:{gy},scale={target_w}:-2:flags={scale_flags}[game];"
            f"[bg][cam]overlay=0:0:shortest=1[tmp];"
            f"[tmp][game]overlay=0:{ch_out + 40}:shortest=1[v_pre]"
        )

    else:
        gh_out = to_even(target_h * 0.60)
        ch_out = target_h - gh_out
        filter_str = (
            f"[0:v]crop={gw}:{gh}:{gx}:{gy},scale={target_w}:{gh_out}:force_original_aspect_ratio=increase:flags={scale_flags},crop={target_w}:{gh_out}[top];"
            f"[0:v]crop={cw}:{ch}:{cx}:{cy},scale={target_w}:{ch_out}:force_original_aspect_ratio=increase:flags={scale_flags},crop={target_w}:{ch_out}[bot];"
            f"[top][bot]vstack=inputs=2[v_pre]"
        )

    filter_str = f"{filter_str};[v_pre]fps={fps_val},setsar=1[v_base]"
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


def compute_caption_font_px(font_size_sp: Any = 19, canvas_w: int = 1080) -> int:
    """Maps UI caption size (14..32) to exact proportional pixels on [canvas_w] (19 -> 57px on 1080w)."""
    try:
        sp = float(font_size_sp)
    except Exception:
        sp = 19.0
    clamped = max(14.0, min(32.0, sp))
    return max(24, int(round(clamped * (canvas_w / 360.0))))


def hex_to_ass_color(hex_str: str, alpha_byte: int = 0) -> str:
    """Converts #RRGGBB into ASS &HAABBGGRR color string."""
    r, g, b = parse_hex_color(hex_str, default=(243, 107, 22))
    a = max(0, min(255, int(alpha_byte)))
    return f"&H{a:02X}{b:02X}{g:02X}{r:02X}"


def format_srt(transcript: list[dict], clip_start: float, clip_end: float) -> str:
    def _stamp(seconds: float) -> str:
        ms = max(0, int(round(seconds * 1000)))
        hh, rem = divmod(ms, 3600000)
        mm, rem = divmod(rem, 60000)
        ss, milli = divmod(rem, 1000)
        return f"{hh:02d}:{mm:02d}:{ss:02d},{milli:03d}"

    rows = []
    for seg in transcript:
        a = max(float(clip_start), float(seg.get("start", 0.0)))
        b = min(float(clip_end), float(seg.get("end", 0.0)))
        txt = str(seg.get("text") or "").strip()
        if b > a and txt:
            rows.append((a - clip_start, b - clip_start, txt))
    return "\n\n".join(f"{n}\n{_stamp(a)} --> {_stamp(b)}\n{txt}" for n, (a, b, txt) in enumerate(rows, 1))


def format_ass(
    transcript: list[dict],
    clip_start: float,
    clip_end: float,
    font_family: str = "Montserrat",
    color_hex: str = "#B7F36B",
    position: str = "center",
    animation: str = "karaoke_pop",
    font_size_sp: Any = 19,
    canvas_w: int = 1080,
    canvas_h: int = 1920,
) -> str:
    """Builds an Advanced SubStation Alpha (.ass) file matching PlayResX/Y, Fontsize, and style."""
    norm_anim = normalize_caption_animation(animation)
    font_px = compute_caption_font_px(font_size_sp, canvas_w)
    primary_ass = hex_to_ass_color(color_hex) if norm_anim != "clean_cut" else "&H00F7F9FA"
    secondary_ass = "&H00F7F9FA"
    outline_ass = "&H000C0C0C"
    back_ass = "&H80000000"
    align = 8 if position == "top" else (2 if position == "bottom" else 5)
    margin_v = int(canvas_h * 0.11) if position == "top" else (int(canvas_h * 0.16) if position == "bottom" else 40)

    def _ass_ts(seconds: float) -> str:
        cs_total = max(0, int(round(seconds * 100)))
        hh, rem = divmod(cs_total, 360000)
        mm, rem = divmod(rem, 6000)
        ss, cs = divmod(rem, 100)
        return f"{hh}:{mm:02d}:{ss:02d}.{cs:02d}"

    header = (
        "[Script Info]\n"
        "Title: FastClip Exported Subtitles\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {canvas_w}\n"
        f"PlayResY: {canvas_h}\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font_family},{font_px},{primary_ass},{secondary_ass},{outline_ass},{back_ass},"
        f"-1,0,0,0,100,100,0,0,1,4,2,{align},48,48,{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    cues = _build_word_chunks(transcript, clip_start, clip_end)
    events = []
    if cues:
        for cue in cues:
            s_t = _ass_ts(cue["start"])
            e_t = _ass_ts(cue["end"])
            if norm_anim == "clean_cut":
                txt = " ".join(w["word"] for w in cue["words"])
            else:
                parts = []
                for w in cue["words"]:
                    dur_cs = max(5, int(round((float(w["end"]) - float(w["start"])) * 100)))
                    parts.append(f"{{\\kf{dur_cs}}}{w['word']}")
                txt = " ".join(parts)
            events.append(f"Dialogue: 0,{s_t},{e_t},Default,,0,0,0,,{txt}")
    else:
        for seg in transcript:
            a = max(float(clip_start), float(seg.get("start", 0.0)))
            b = min(float(clip_end), float(seg.get("end", 0.0)))
            txt = str(seg.get("text") or "").strip()
            if b > a and txt:
                events.append(f"Dialogue: 0,{_ass_ts(a - clip_start)},{_ass_ts(b - clip_start)},Default,,0,0,0,,{txt}")
    return header + "\n".join(events) + ("\n" if events else "")


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
    out_mov: Path,
    font_size_sp: Any = 19
) -> tuple[bool, int, int, int]:
    """
    Renders a smooth 60fps transparent QuickTime RLE (.mov) caption strip using
    word-level timestamps, elastic spring/cubic easing, and single-overlay compositing.
    Returns (created, strip_render_w, strip_render_h, band_y_in_target).
    """
    cues = _build_word_chunks(transcript, clip_start, clip_end)
    if not cues:
        return False, 0, 0, 0
    animation = normalize_caption_animation(animation)

    dur = max(0.5, clip_end - clip_start)
    render_w = min(target_w, 1080)
    base_size = compute_caption_font_px(font_size_sp, render_w)
    pop_size = max(base_size + 4, int(round(base_size * 1.14)))
    small_size = max(20, int(round(base_size * 0.80)))
    raw_h = max(260, int(round(base_size * 3.6)))
    render_h = raw_h + (raw_h % 2)
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
    def _load_heavy_font(sz: int):
        cache_key = (str(font_file), sz)
        cached_f = _FONT_CACHE.get(cache_key)
        if cached_f is not None:
            return cached_f
        f = ImageFont.truetype(str(font_file), sz)
        try:
            f.set_variation_by_axes([850])
        except Exception:
            pass
        _FONT_CACHE[cache_key] = f
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
                is_act = (wi == active_idx) and (animation != "clean_cut")
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

            # Clean dark pill backing for standard (clean_cut) readability scaled proportionally with font size
            if animation == "clean_cut":
                pad_x = max(24, int(round(base_size * 0.52)))
                box_half_h = max(32, int(round(base_size * 0.88)))
                pill_a = int(205 * alpha)
                border_a = int(45 * alpha)
                x0 = render_w // 2 - total_w // 2 - pad_x
                x1 = render_w // 2 + total_w // 2 + pad_x
                y0 = cy - box_half_h
                y1 = cy + box_half_h
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
    finally:
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            try:
                if pipe and not pipe.closed:
                    pipe.close()
            except Exception:
                pass


HIGHLIGHT_CUES = (
    "wow", "oh", "no way", "amazing", "insane", "incredible", "finally",
    "let's go", "laugh", "unbelievable", "what", "why", "clutch", "op", "bhai", "kya", "arre"
)


def score_transcript_segment(seg: dict) -> float:
    """Scores a transcript segment by speech density, hype/reaction cues, and exclamation intensity."""
    text = str(seg.get("text") or "").lower()
    score = min(len(text) / 90.0, 1.5)
    score += sum(1.0 for cue in HIGHLIGHT_CUES if cue in text)
    score += 0.35 * text.count("!") + 0.2 * text.count("?")
    return round(score, 4)


def select_ranked_highlights(
    transcript: list[dict],
    duration: float,
    max_highlights: Any = 5,
    pad_seconds: float = 15.0,
    min_clip_seconds: float = 6.0,
) -> list[tuple[float, float, float, str]]:
    """
    Ranks transcript segments by excitement score descending and returns up to
    clamp(max_highlights, 1, 20) non-overlapping (score, start, end, label) tuples.
    Returns fewer if fewer non-overlapping segments are available; falls back to
    the opening segment when transcript yields no valid candidates.
    """
    try:
        cap = max(1, min(int(max_highlights), 20))
    except Exception:
        cap = 5

    ranked: list[tuple[float, float, float, str]] = []
    for s in transcript:
        start_s = float(s.get("start", 0.0))
        end_s = float(s.get("end", start_s))
        score = float(s["score"]) if "score" in s else score_transcript_segment(s)
        a = max(0.0, start_s - pad_seconds)
        b = min(float(duration), end_s + pad_seconds)
        if b - a >= min_clip_seconds:
            ranked.append((score, a, b, str(s.get("text") or "Highlight")))

    ranked.sort(key=lambda x: x[0], reverse=True)
    chosen: list[tuple[float, float, float, str]] = []
    for score, a, b, label in ranked:
        if all(b <= x[1] or a >= x[2] for x in chosen):
            chosen.append((score, a, b, label))
        if len(chosen) >= cap:
            break

    if not chosen and duration > 0:
        chosen = [(0.0, 0.0, min(float(duration), 60.0), "Opening segment")]
    return chosen


KNOWN_GAMES = [
    ("valorant", "Valorant"),
    ("gta v", "GTAV"),
    ("grand theft auto", "GTAV"),
    ("bgmi", "BGMI"),
    ("pubg", "PUBG"),
    ("minecraft", "Minecraft"),
    ("counter-strike", "CS2"),
    ("cs2", "CS2"),
    ("cs:go", "CSGO"),
    ("apex legends", "ApexLegends"),
    ("fortnite", "Fortnite"),
    ("free fire", "FreeFire"),
    ("call of duty", "COD"),
    ("warzone", "Warzone"),
    ("elden ring", "EldenRing"),
    ("league of legends", "LoL"),
    ("dota 2", "Dota2"),
    ("overwatch", "Overwatch"),
    ("rocket league", "RocketLeague"),
    ("fifa", "FIFA"),
    ("ea sports fc", "EAFC"),
    ("schedule i", "ScheduleI"),
    ("repo", "REPO"),
]


def sanitize_name_token(raw: Any, fallback: str = "Clip") -> str:
    """Strips filesystem-unsafe/special chars cleanly into an alphanumeric Pascal-ish token."""
    s = str(raw or "").strip()
    if not s:
        return fallback
    parts = re.split(r"[^a-zA-Z0-9]+", s)
    tokens = [p for p in parts if p]
    if not tokens:
        return fallback
    # Preserve casing for uppercase acronyms (BGMI, GTA, FPS), capitalize words otherwise
    cleaned = "".join(t if (t.isupper() or any(c.isupper() for c in t[1:])) else t.capitalize() for t in tokens)
    return cleaned[:28] or fallback


def normalize_res_label(resolution: Any) -> str:
    r = str(resolution or "1080p").strip().lower()
    if r == "4k":
        return "4K"
    if r == "2k":
        return "2K"
    if r.endswith("p"):
        return r
    return sanitize_name_token(r, "1080p")


def infer_game_and_streamer(meta: dict) -> tuple[str, str]:
    """Best-effort inference of (Game, Streamer) from yt-dlp or oEmbed JSON metadata."""
    raw_streamer = (
        meta.get("channel")
        or meta.get("uploader")
        or meta.get("creator")
        or meta.get("author_name")
        or ""
    )
    streamer = sanitize_name_token(raw_streamer, fallback="Streamer")

    explicit_game = meta.get("game") or meta.get("track") or ""
    if explicit_game:
        return sanitize_name_token(explicit_game, fallback="Gameplay"), streamer

    haystack = " ".join(
        [
            str(meta.get("title") or ""),
            " ".join(str(x) for x in (meta.get("categories") or [])),
            " ".join(str(x) for x in (meta.get("tags") or [])[:15]),
        ]
    ).lower()
    for needle, canonical in KNOWN_GAMES:
        if needle in haystack:
            return canonical, streamer
    return "Gameplay", streamer


def build_clip_filename(
    game: Any,
    streamer: Any,
    resolution: Any,
    existing_names: list[str] | tuple[str, ...] | set[str] = (),
) -> str:
    """
    Formats `{Game}_{Streamer}_{Resolution}_V{n}.mp4`, auto-incrementing n safely
    across any existing names/files with the same prefix to prevent collisions.
    """
    g = sanitize_name_token(game, fallback="Gameplay")
    st = sanitize_name_token(streamer, fallback="Streamer")
    res = normalize_res_label(resolution)
    prefix = f"{g}_{st}_{res}_V"
    lower_existing = {Path(str(name)).stem.lower() for name in existing_names if str(name).strip()}
    pattern = re.compile(rf"^{re.escape(prefix.lower())}(\d+)$")
    max_v = 0
    for stem in lower_existing:
        m = pattern.match(stem)
        if m:
            max_v = max(max_v, int(m.group(1)))
    next_v = max_v + 1
    while f"{prefix.lower()}{next_v}" in lower_existing:
        next_v += 1
    return f"{prefix}{next_v}.mp4"


def _parse_fps_fraction(raw_rate: Any) -> float:
    txt = str(raw_rate or "").strip()
    if "/" in txt:
        num, den = txt.split("/", 1)
        try:
            d = float(den)
            return round(float(num) / d, 2) if d > 0 else 0.0
        except Exception:
            return 0.0
    try:
        return round(float(txt), 2)
    except Exception:
        return 0.0


def parse_ffprobe_json(
    raw: dict,
    source_url: str = "",
    render_time_s: float | None = None,
) -> dict:
    """Parses ffprobe -show_format -show_streams -of json output into structured Video Info."""
    streams = raw.get("streams") or []
    fmt = raw.get("format") or {}
    v_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    a_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})

    w = int(v_stream.get("width") or 0)
    h = int(v_stream.get("height") or 0)
    fps = _parse_fps_fraction(v_stream.get("avg_frame_rate") or v_stream.get("r_frame_rate"))
    if fps <= 0:
        fps = _parse_fps_fraction(v_stream.get("r_frame_rate"))

    dur_s = round(safe_float(fmt.get("duration") or v_stream.get("duration"), 0.0), 2)
    size_bytes = int(safe_float(fmt.get("size"), 0.0))
    bit_rate_bps = safe_float(fmt.get("bit_rate") or v_stream.get("bit_rate"), 0.0)
    bitrate_kbps = int(round(bit_rate_bps / 1000.0)) if bit_rate_bps > 0 else 0

    return {
        "url": source_url,
        "width": w,
        "height": h,
        "resolution": f"{w}x{h}" if w > 0 and h > 0 else "",
        "fps": fps,
        "duration_s": dur_s,
        "video_codec": str(v_stream.get("codec_name") or ""),
        "audio_codec": str(a_stream.get("codec_name") or ""),
        "bitrate_kbps": bitrate_kbps,
        "size_bytes": size_bytes,
        "render_time_s": round(float(render_time_s), 2) if render_time_s is not None else 0.0,
    }


def probe_media_info(file_path: Path, source_url: str = "", render_time_s: float | None = None) -> dict:
    """Executes ffprobe JSON inspection directly on [file_path]."""
    try:
        raw_out = run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(file_path)],
            timeout=15,
        )
        parsed = parse_ffprobe_json(json.loads(raw_out), source_url=source_url, render_time_s=render_time_s)
        if parsed["size_bytes"] <= 0 and file_path.is_file():
            parsed["size_bytes"] = file_path.stat().st_size
        return parsed
    except Exception as pe:
        logger.warning(f"ffprobe JSON failed on {file_path}: {pe}")
        return {
            "url": source_url,
            "width": 0,
            "height": 0,
            "resolution": "",
            "fps": 0.0,
            "duration_s": 0.0,
            "video_codec": "",
            "audio_codec": "",
            "bitrate_kbps": 0,
            "size_bytes": file_path.stat().st_size if file_path.is_file() else 0,
            "render_time_s": round(float(render_time_s), 2) if render_time_s is not None else 0.0,
        }


def extract_audio_pcm(file_path: Path) -> np.ndarray:
    """Decodes media file audio to 16kHz float32 mono array via FFmpeg."""
    cmd = ["ffmpeg", "-y", "-i", str(file_path), "-f", "s16le", "-ac", "1", "-ar", "16000", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError("Failed to decode audio with ffmpeg")
    return np.frombuffer(p.stdout, np.int16).flatten().astype(np.float32) / 32768.0


CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_LOCK = threading.Lock()
DEFAULT_CACHE_LIMIT_BYTES = int(os.getenv("FASTCLIP_CACHE_LIMIT_MB", "4096")) * 1024 * 1024
CACHE_MAX_AGE_SECONDS = int(os.getenv("FASTCLIP_CACHE_MAX_AGE_DAYS", "7")) * 86400


def _cache_index_path(cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / "sources.json"


def _load_cache_index(cache_dir: Path = CACHE_DIR) -> dict:
    idx = _cache_index_path(cache_dir)
    if idx.exists():
        try:
            data = json.loads(idx.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("entries"), dict):
                return data
        except Exception:
            pass
    return {"entries": {}}


def _save_cache_index(data: dict, cache_dir: Path = CACHE_DIR) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / "sources.json.tmp"
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_cache_index_path(cache_dir))


def _prune_missing_and_expired(data: dict, max_age_seconds: int = CACHE_MAX_AGE_SECONDS) -> dict:
    now = time.time()
    kept = {}
    for sid, entry in (data.get("entries") or {}).items():
        if not isinstance(entry, dict):
            continue
        sections = []
        for sec in entry.get("sections") or []:
            sp = Path(sec.get("path", ""))
            if sp.is_file() and (now - float(sec.get("last_used", now))) <= max_age_seconds:
                sec["size_bytes"] = sp.stat().st_size
                sections.append(sec)
            elif sp.is_file():
                sp.unlink(missing_ok=True)
        fp = Path(entry.get("full_path", "")) if entry.get("full_path") else None
        has_full = fp is not None and fp.is_file()
        if has_full and (now - float(entry.get("last_used", now))) > max_age_seconds:
            fp.unlink(missing_ok=True)
            has_full = False
        if has_full or sections:
            entry["full_path"] = str(fp) if has_full else ""
            entry["sections"] = sections
            entry["size_bytes"] = (fp.stat().st_size if has_full else 0) + sum(s["size_bytes"] for s in sections)
            kept[sid] = entry
    data["entries"] = kept
    return data


def prune_cache_lru(cache_dir: Path = CACHE_DIR, max_bytes: int = DEFAULT_CACHE_LIMIT_BYTES) -> dict:
    with CACHE_LOCK:
        data = _prune_missing_and_expired(_load_cache_index(cache_dir))
        # Build flat list of artifacts (last_used, kind, stream_id, path, size)
        items = []
        total = 0
        for sid, entry in data["entries"].items():
            if entry.get("full_path"):
                p = Path(entry["full_path"])
                if p.is_file():
                    sz = p.stat().st_size
                    total += sz
                    items.append((float(entry.get("last_used", 0)), "full", sid, p, sz))
            for sec in entry.get("sections", []):
                p = Path(sec["path"])
                if p.is_file():
                    sz = p.stat().st_size
                    total += sz
                    items.append((float(sec.get("last_used", 0)), "section", sid, p, sz))
        items.sort(key=lambda x: x[0])  # oldest first
        for _, kind, sid, p, sz in items:
            if total <= max_bytes:
                break
            p.unlink(missing_ok=True)
            total -= sz
            entry = data["entries"].get(sid, {})
            if kind == "full":
                entry["full_path"] = ""
            else:
                entry["sections"] = [s for s in entry.get("sections", []) if s.get("path") != str(p)]
        data = _prune_missing_and_expired(data)
        _save_cache_index(data, cache_dir)
        return data


def get_cache_stats(cache_dir: Path = CACHE_DIR, limit_bytes: int = DEFAULT_CACHE_LIMIT_BYTES) -> dict:
    with CACHE_LOCK:
        data = _prune_missing_and_expired(_load_cache_index(cache_dir))
        _save_cache_index(data, cache_dir)
        entries_out = []
        total = 0
        for sid, entry in data["entries"].items():
            sz = int(entry.get("size_bytes", 0))
            total += sz
            entries_out.append({
                "stream_id": sid,
                "url": entry.get("url", ""),
                "title": entry.get("title", sid),
                "size_bytes": sz,
                "last_used": float(entry.get("last_used", 0)),
                "has_full": bool(entry.get("full_path")),
                "sections_count": len(entry.get("sections", [])),
            })
        entries_out.sort(key=lambda e: e["last_used"], reverse=True)
        return {"total_bytes": total, "limit_bytes": limit_bytes, "entries": entries_out}


def clear_source_cache(cache_dir: Path = CACHE_DIR, stream_id: str | None = None) -> dict:
    with CACHE_LOCK:
        data = _load_cache_index(cache_dir)
        cleared = 0
        if stream_id:
            entry = data.get("entries", {}).pop(stream_id, None)
            if entry:
                if entry.get("full_path"):
                    Path(entry["full_path"]).unlink(missing_ok=True)
                for sec in entry.get("sections", []):
                    Path(sec.get("path", "")).unlink(missing_ok=True)
                cleared = 1
        else:
            for p in cache_dir.glob("*.mp4"):
                p.unlink(missing_ok=True)
                cleared += 1
            data["entries"] = {}
        _save_cache_index(data, cache_dir)
        return {"cleared": cleared, "total_bytes": 0}


def resolve_cached_source(
    stream_id: str,
    url: str,
    start_seconds: float | None,
    end_seconds: float | None,
    work_source_out: Path,
    downloader_fn,
    cache_dir: Path = CACHE_DIR,
    max_cache_bytes: int = DEFAULT_CACHE_LIMIT_BYTES,
    title: str = "",
) -> tuple[Path, bool]:
    """
    Returns (source_file_path, cache_hit).
    Reuses locally cached full VODs or cached section segments without triggering [downloader_fn].
    Enforces [max_cache_bytes] LRU retention on cache writes.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    with CACHE_LOCK:
        data = _prune_missing_and_expired(_load_cache_index(cache_dir))
        entry = data["entries"].get(stream_id, {})
        now = time.time()

        # 1. Check cached full stream
        full_p = Path(entry.get("full_path", "")) if entry.get("full_path") else (cache_dir / f"{stream_id}.mp4")
        if full_p.is_file():
            try:
                if start_seconds is None or end_seconds is None:
                    shutil.copy2(str(full_p), str(work_source_out))
                    entry["last_used"] = now
                    data["entries"][stream_id] = entry
                    _save_cache_index(data, cache_dir)
                    return work_source_out, True
                cdur = float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(full_p)], timeout=10))
                if cdur >= float(end_seconds) - 0.25:
                    run(["ffmpeg", "-y", "-ss", str(start_seconds), "-i", str(full_p), "-t", str(max(0.5, end_seconds - start_seconds)), "-c", "copy", str(work_source_out)], timeout=30)
                    entry["last_used"] = now
                    data["entries"][stream_id] = entry
                    _save_cache_index(data, cache_dir)
                    return work_source_out, True
            except Exception as ce:
                logger.warning(f"Full cache slice check failed: {ce}")

        # 2. Check cached section segments covering [start_seconds, end_seconds]
        if start_seconds is not None and end_seconds is not None:
            for sec in entry.get("sections", []):
                sp = Path(sec.get("path", ""))
                s_a, s_b = float(sec.get("start", 0.0)), float(sec.get("end", 0.0))
                if sp.is_file() and s_a <= float(start_seconds) + 0.05 and s_b >= float(end_seconds) - 0.05:
                    offset = max(0.0, float(start_seconds) - s_a)
                    dur_s = max(0.5, float(end_seconds) - float(start_seconds))
                    try:
                        if offset < 0.05 and abs(dur_s - (s_b - s_a)) < 0.15:
                            shutil.copy2(str(sp), str(work_source_out))
                        else:
                            run(["ffmpeg", "-y", "-ss", str(offset), "-i", str(sp), "-t", str(dur_s), "-c", "copy", str(work_source_out)], timeout=30)
                        sec["last_used"] = now
                        entry["last_used"] = now
                        _save_cache_index(data, cache_dir)
                        return work_source_out, True
                    except Exception as se:
                        logger.warning(f"Section cache extract notice: {se}")

    # 3. Cache miss -> run downloader_fn once and persist into cache_dir
    downloaded = Path(downloader_fn())
    if not downloaded.is_file():
        raise RuntimeError("Downloader produced no source file")

    with CACHE_LOCK:
        data = _prune_missing_and_expired(_load_cache_index(cache_dir))
        now = time.time()
        entry = data["entries"].setdefault(stream_id, {"url": url, "title": title or stream_id, "full_path": "", "sections": [], "last_used": now, "size_bytes": 0})
        if title:
            entry["title"] = title
        entry["url"] = url or entry.get("url", "")
        entry["last_used"] = now
        try:
            if start_seconds is None or end_seconds is None:
                dest = cache_dir / f"{stream_id}.mp4"
                shutil.copy2(str(downloaded), str(dest))
                entry["full_path"] = str(dest)
            else:
                dest = cache_dir / f"{stream_id}__{int(round(start_seconds))}_{int(round(end_seconds))}.mp4"
                shutil.copy2(str(downloaded), str(dest))
                sections = [s for s in entry.get("sections", []) if s.get("path") != str(dest)]
                sections.append({
                    "start": float(start_seconds),
                    "end": float(end_seconds),
                    "path": str(dest),
                    "size_bytes": dest.stat().st_size,
                    "last_used": now,
                })
                entry["sections"] = sections
        except Exception as ce:
            logger.warning(f"Cache persist warning: {ce}")
        _save_cache_index(data, cache_dir)

    prune_cache_lru(cache_dir, max_cache_bytes)
    return downloaded, False


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
        work_source = work / "source.mp4"

        def _download_from_source() -> Path:
            ytdlp_cmd = build_ytdlp_base_cmd()
            ytdlp_cmd.extend([
                "-S", "res,fps,vcodec:h264",
                "-f", "bestvideo[fps>=50]+bestaudio/bestvideo+bestaudio/best",
                "--merge-output-format", "mp4",
                "-o", str(work / "source.%(ext)s")
            ])

            if req.start_seconds is not None and req.end_seconds is not None:
                ytdlp_cmd.extend(["--download-sections", f"*{req.start_seconds}-{req.end_seconds}"])
            else:
                ytdlp_cmd.append("--live-from-start")

            ytdlp_cmd.append(clean_url)
            try:
                run(ytdlp_cmd, timeout=240)
            except Exception as dl_err:
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
            found = [p for p in work.glob("source.*") if p.is_file()]
            if not found:
                raise RuntimeError("Downloader produced no source file")
            return found[0]

        source, cache_hit = resolve_cached_source(
            stream_id=stream_id,
            url=clean_url,
            start_seconds=safe_float(req.start_seconds, -1.0) if req.start_seconds is not None else None,
            end_seconds=safe_float(req.end_seconds, -1.0) if req.end_seconds is not None else None,
            work_source_out=work_source,
            downloader_fn=_download_from_source,
        )
        t_dl = (datetime.now() - t_start).total_seconds()
        update(cache_hit=cache_hit, progress=28)

        caption_lang = normalize_caption_language(req.language, req.captions_enabled)
        need_auto_highlights = (req.start_seconds is None or req.end_seconds is None)
        need_audio = (caption_lang != "off" or need_auto_highlights)

        # Parallelize ffprobe source inspection and 16kHz PCM audio extraction
        audio_pcm = np.array([], dtype=np.float32)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as prep_pool:
            fut_probe = prep_pool.submit(probe_media_info, source, clean_url)
            fut_audio = prep_pool.submit(extract_audio_pcm, source) if need_audio else None
            source_info = fut_probe.result()
            if fut_audio is not None:
                try:
                    audio_pcm = fut_audio.result()
                except Exception as ae:
                    logger.warning(f"Audio extraction notice: {ae}")

        dur = source_info["duration_s"] or float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(source)]))
        if source_info["width"] > 0 and source_info["height"] > 0:
            src_w, src_h = source_info["width"], source_info["height"]
        else:
            dims_str = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", str(source)])
            src_w, src_h = map(int, dims_str.strip().split("x")[:2])

        update(status="transcribing", progress=35, source_duration=dur, source_res=f"{src_w}x{src_h}", source_info=source_info)

        transcript = []
        captions_meta = {"engine": "off", "model": "none", "language": caption_lang, "elapsed_s": 0.0}
        if need_audio and audio_pcm.size > 0:
            try:
                tx_lang = caption_lang if caption_lang != "off" else "en"
                transcript, captions_meta = transcribe_audio_pcm(audio_pcm, caption_language=tx_lang)
            except Exception as we:
                logger.warning(f"Whisper transcription fallback: {we}")
                transcript = []
        (work / "transcript.json").write_text(json.dumps(transcript, ensure_ascii=False, indent=2))

        update(status="selecting", progress=60, transcript_segments=len(transcript), captions_meta=captions_meta)

        if req.start_seconds is not None and req.end_seconds is not None:
            ranges = [(0.0, dur, "Selected clip")]
        else:
            chosen = select_ranked_highlights(transcript, dur, max_highlights=req.max_highlights)
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

        src_fps = float(source_info.get("fps") or 0.0)
        target_fps = max(60, int(round(src_fps))) if src_fps >= 60.0 else 60
        enc_args, enc_name = get_best_video_encoder(
            req.resolution,
            target_fps=target_fps,
            encoder_mode=getattr(req, "encoder_mode", "auto"),
        )

        used_names: list[str] = [p.name for p in work.glob("*.mp4")]
        g_label = str(getattr(req, "game_name", "") or "").strip() or "Gameplay"
        s_label = str(getattr(req, "streamer_name", "") or "").strip() or "Streamer"

        # Pre-generate caption strips ahead of VideoToolbox encoding so multi-clip prep overlaps encode
        cap_futures: dict[int, concurrent.futures.Future] = {}
        cap_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        try:
            for idx_p, (st_p, en_p, _) in enumerate(ranges, 1):
                if req.layout != "original" and caption_lang != "off":
                    srt_txt_p = format_srt(transcript, st_p, en_p)
                    if srt_txt_p:
                        caps_mov_p = work / f"caps_{idx_p:02d}.mov"
                        cap_futures[idx_p] = cap_pool.submit(
                            generate_animated_captions_mov,
                            transcript, st_p, en_p, target_w, target_h,
                            req.caption_font, req.caption_color, req.caption_position,
                            req.caption_animation, caps_mov_p, getattr(req, "caption_font_size", 19),
                        )

            total_clips = max(1, len(ranges))
            for idx, (start, end, label) in enumerate(ranges, 1):
                base_pct = 75 + int((idx - 1) * (20 / total_clips))
                update(progress=base_pct)
                t_clip_0 = time.perf_counter()
                out_filename = build_clip_filename(g_label, s_label, req.resolution, existing_names=used_names)
                used_names.append(out_filename)
                out = work / out_filename
                srt = work / f"{out.stem}.srt"
                ass = work / f"{out.stem}.ass"

                srt_text = format_srt(transcript, start, end)
                srt.write_text(srt_text, encoding="utf-8")
                ass_text = format_ass(
                    transcript=transcript,
                    clip_start=start,
                    clip_end=end,
                    font_family=req.caption_font,
                    color_hex=req.caption_color,
                    position=req.caption_position,
                    animation=req.caption_animation,
                    font_size_sp=getattr(req, "caption_font_size", 19),
                    canvas_w=target_w,
                    canvas_h=target_h,
                )
                ass.write_text(ass_text, encoding="utf-8")

                if req.layout == "original":
                    cmd = ["ffmpeg", "-y", "-ss", str(start), "-i", str(source), "-t", str(end - start),
                           "-c", "copy", "-movflags", "+faststart", str(out)]
                    run(cmd)
                else:
                    base_filter, cur_v = build_9_16_filter(
                        source_w=src_w,
                        source_h=src_h,
                        layout=req.layout,
                        game_size=req.game_size,
                        cam=cam_b,
                        game=game_b,
                        target_w=target_w,
                        target_h=target_h,
                        target_fps=target_fps,
                    )

                    cmd = [
                        "ffmpeg", "-y",
                        "-ss", str(start),
                        "-i", str(source),
                        "-t", str(end - start)
                    ]

                    filter_parts = [base_filter]

                    if idx in cap_futures:
                        caps_mov = work / f"caps_{idx:02d}.mov"
                        ok_caps, strip_w, _, band_y = cap_futures[idx].result()
                        if ok_caps:
                            cmd.extend(["-i", str(caps_mov)])
                            if strip_w != target_w:
                                filter_parts.append(f"[1:v]scale={target_w}:-2:flags=lanczos[cap_scaled]")
                                filter_parts.append(f"{cur_v}[cap_scaled]overlay=0:{band_y}:shortest=1[v_final]")
                            else:
                                filter_parts.append(f"{cur_v}[1:v]overlay=0:{band_y}:shortest=1[v_final]")
                            cur_v = "[v_final]"

                    final_filter = ";".join(filter_parts)

                    def _build_encode_cmd(v_args: list[str]) -> list[str]:
                        return [
                            *cmd,
                            "-filter_complex", final_filter,
                            "-map", cur_v,
                            "-map", "0:a:0?",
                            *v_args,
                            "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
                            "-c:a", "aac",
                            "-b:a", "320k",
                            "-ar", "48000",
                            "-movflags", "+faststart",
                            str(out),
                        ]

                    try:
                        run(_build_encode_cmd(enc_args), timeout=600)
                    except Exception as enc_err:
                        if "videotoolbox" in enc_name:
                            logger.warning(f"GPU VideoToolbox render failed, falling back to CPU libx264: {enc_err}")
                            fb_args, enc_name = get_best_video_encoder(req.resolution, target_fps=target_fps, encoder_mode="cpu")
                            run(_build_encode_cmd(fb_args), timeout=600)
                        else:
                            raise

                clip_render_s = round(time.perf_counter() - t_clip_0, 2)
                output_info = probe_media_info(out, render_time_s=clip_render_s)
                outputs.append({
                    "name": out.name,
                    "label": label,
                    "start": start,
                    "end": end,
                    "download": f"/jobs/{job_id}/files/{out.name}",
                    "captions": f"/jobs/{job_id}/files/{srt.name}",
                    "captions_ass": f"/jobs/{job_id}/files/{ass.name}",
                    "captions_meta": captions_meta,
                    "source_info": source_info,
                    "output_info": output_info,
                })
        finally:
            cap_pool.shutdown(wait=False)

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


@app.on_event("startup")
def on_startup():
    warmup_runtime_async()


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

    source_info = {
        "url": clean_url,
        "width": 1920,
        "height": 1080,
        "resolution": "1920x1080",
        "fps": 0.0,
        "duration_s": 0.0,
        "video_codec": "",
        "audio_codec": "",
        "bitrate_kbps": 0,
        "size_bytes": 0,
        "render_time_s": 0.0,
    }
    inferred_game = "Gameplay"
    inferred_streamer = "Streamer"
    try:
        ytdlp_cmd = build_ytdlp_base_cmd() + ["--dump-json", clean_url]
        raw_json = run(ytdlp_cmd, timeout=45)
        meta = json.loads(raw_json)
        title = meta.get("title", "Livestream")
        duration = float(meta.get("duration") or 0.0)
        web_thumb = meta.get("thumbnail") or ""
        src_w = int(meta.get("width") or 1920)
        src_h = int(meta.get("height") or 1080)
        inferred_game, inferred_streamer = infer_game_and_streamer(meta)
        source_info = {
            "url": clean_url,
            "width": src_w,
            "height": src_h,
            "resolution": f"{src_w}x{src_h}",
            "fps": round(safe_float(meta.get("fps"), 0.0), 2),
            "duration_s": round(duration, 2),
            "video_codec": str(meta.get("vcodec") or "").split(".")[0],
            "audio_codec": str(meta.get("acodec") or "").split(".")[0],
            "bitrate_kbps": int(round(safe_float(meta.get("tbr"), 0.0))),
            "size_bytes": int(safe_float(meta.get("filesize") or meta.get("filesize_approx"), 0.0)),
            "render_time_s": 0.0,
        }
    except Exception as e:
        logger.warning(f"yt-dlp metadata fetch failed, falling back to oEmbed: {e}")
        oembed = fetch_youtube_oembed(clean_url)
        title = oembed.get("title") or "Livestream"
        duration = 0.0
        web_thumb = oembed.get("thumbnail_url") or f"https://img.youtube.com/vi/{stream_id}/hqdefault.jpg"
        src_w, src_h = 1920, 1080
        source_info["url"] = clean_url
        inferred_game, inferred_streamer = infer_game_and_streamer(oembed)

    cached_full = CACHE_DIR / f"{stream_id}.mp4"
    if cached_full.is_file():
        source_info = probe_media_info(cached_full, source_url=clean_url)

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
        "default_game_box": game_box,
        "source_info": source_info,
        "inferred_game": inferred_game,
        "inferred_streamer": inferred_streamer,
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


@app.get("/cache")
def read_cache():
    return get_cache_stats()


@app.delete("/cache")
def delete_cache(stream_id: str | None = None):
    return clear_source_cache(stream_id=stream_id)


@app.api_route("/jobs/{job_id}/files/{filename}", methods=["GET", "HEAD"])
def get_file(job_id: str, filename: str):
    path = (ROOT / job_id / filename).resolve()
    if path.parent != (ROOT / job_id).resolve() or not path.is_file() or path.suffix.lower() not in (".mp4", ".srt", ".ass"):
        raise HTTPException(404, "File not found")
    media_type = "video/mp4" if path.suffix.lower() == ".mp4" else "text/plain; charset=utf-8"
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

